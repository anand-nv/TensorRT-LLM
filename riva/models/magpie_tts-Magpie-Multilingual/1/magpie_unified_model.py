# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import base64
import datetime
import json
import logging
import math
import os
import queue
import random
import re
import signal
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch
import triton_python_backend_utils as pb_utils
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.runtime import ModelConfig, ModelRunnerCpp, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from torch.utils.dlpack import from_dlpack, to_dlpack

# Using threading instead of multiprocessing for better resource sharing


QUEUE_TIMEOUT = 0.010
AUDIO_QUEUE_TIMEOUT = 0.020  # Longer timeout for audio processing to avoid race conditions


class RequestStatus(Enum):
    """
    RequestStatus is an enum that represents the status of a request.
    """

    IN_PROGRESS = 0
    CONCLUDING = 1
    COMPLETED = 2
    FAILED = 3
    INIT = 4
    WORK_DONE = 5


class AudioDecoderWorkItem:
    # Shared crossfade buffers — initialized lazily, keyed by (sample_rate, fade_samples).
    _fade_cache: dict = {}

    def __init__(
        self,
        corr_id,
        chunk_size,
        history_size,
        future_size,
        num_audio_codebooks,
        book_size,
        eos_token,
        max_tokens,
        trtllm_request_id,
        response_sender,
        sentence_num,
        is_last_sentence,
        downsampling_factor,
        logger,
        audio_sample_rate,
        is_last_request,
        stacking_factor=1,
        fade_ms=5,
        read_idx=0,
        write_idx=0,
        audio_tokens=None,
    ):
        self.audio_codes = torch.zeros((num_audio_codebooks, max_tokens))
        self.corr_id = corr_id
        self.history_size = history_size
        self.future_size = future_size
        self.num_audio_codebooks = num_audio_codebooks
        self.stacking_factor = stacking_factor
        self.chunk_size = chunk_size
        self.read_idx = read_idx
        self.write_idx = write_idx
        self.book_size = book_size
        self.eos_token = eos_token
        self.token_lock = threading.RLock()
        self.trtllm_request_id = trtllm_request_id
        self.response_sender = response_sender
        self.sentence_num = sentence_num
        self.truncated_audio = False
        self.is_last_sentence = is_last_sentence
        self.status = RequestStatus.INIT
        self.logger = logger
        self.sample_rate = audio_sample_rate
        self.downsampling_factor = downsampling_factor
        self.is_last_request = is_last_request

        # Crossfade parameters
        self.fade_ms = fade_ms  # Default sample rate, should be configurable
        self.fade_samples = int(self.sample_rate * self.fade_ms / 1000)
        # Fade curves are static; reuse cached buffers shared across all work items.
        # Avoids torch.linspace + .to("cpu") allocation on every audio chunk init.
        fade_key = (self.sample_rate, self.fade_samples)
        cache = AudioDecoderWorkItem._fade_cache
        if fade_key not in cache:
            cache[fade_key] = (
                torch.linspace(0, 1, self.fade_samples),
                torch.linspace(1, 0, self.fade_samples),
            )
        self.fade_in, self.fade_out = cache[fade_key]

        # Audio buffer for crossfade
        self.future_audio = None

        # Cleanup tracking
        self.send_final_audio = False
        self.is_last_token_in = False
        self.last_token_id = -1
        self.is_last_token_in_read = False
        self.finished = False

        if read_idx > 0 and audio_tokens is not None and audio_tokens.shape[1] > 0:
            self.audio_codes[:, :read_idx] = audio_tokens
        # self.eos_tokens = torch.tensor(
        #     [[self.eos_token] * self.num_audio_codebooks],
        #     dtype=self.audio_codes.dtype,
        #     device=self.audio_codes.device,
        # ).reshape(self.num_audio_codebooks, 1)
        self.eos_tokens = torch.tensor(
            [[621, 1455, 1184, 1038, 463, 377, 1536, 1742]],
            dtype=self.audio_codes.dtype,
            device=self.audio_codes.device,
        ).reshape(self.num_audio_codebooks, 1)

    def apply_crossfade(self, audio_chunk, history_length, finished):
        """
        Apply crossfade to the audio chunk to reduce artifacts between chunks.

        Args:
            audio_chunk: Current audio chunk as torch tensor

        Returns:
            Processed audio chunk with crossfade applied
        """
        history_samples = history_length * self.downsampling_factor
        future_samples = self.future_size * self.downsampling_factor
        if len(audio_chunk.shape) == 1:
            audio_chunk = audio_chunk.unsqueeze(0)
        try:
            if self.future_audio is None:
                self.future_audio = audio_chunk[:, history_samples : audio_chunk.shape[1] - future_samples]
                if finished:
                    return self.future_audio
                return None
            else:
                curr_audio = self.future_audio
                future_audio = audio_chunk[
                    :, history_samples - self.fade_samples : audio_chunk.shape[1] - future_samples + self.fade_samples
                ]
                max_fade_in = min(self.fade_samples, future_audio.shape[1])
                future_audio[:, :max_fade_in] = future_audio[:, :max_fade_in] * self.fade_in[:max_fade_in]
                max_fade_out = min(self.fade_samples, curr_audio.shape[1])
                if max_fade_out > 0:
                    curr_audio[:, -max_fade_out:] = (
                        curr_audio[:, -max_fade_out:] * self.fade_out[:max_fade_out] + future_audio[:, :max_fade_out]
                    )
                self.future_audio = future_audio[:, max_fade_in:]
                self.future_audio = self.future_audio[:, : self.future_audio.shape[1] - max_fade_out]
                return curr_audio
        except Exception as e:
            self.logger.log_error(f"Error {e}")
            raise e

    def process_trtllm_response(self, response):
        # Past EOS — reject any further executor tokens to prevent post-EOS audio artifacts.
        if self.is_last_token_in:
            return True, self.corr_id

        if response is not None:
            result = response.result
            token_ids = result.output_token_ids[0]
            is_final = result.is_final

            if response.has_error():
                triton_error_response = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(response.error)))
                self.logger.log_error(f"Error in process_trtllm_response: {response.error}")
                self.status = RequestStatus.FAILED
                self.response_sender.send(triton_error_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                return False, self.corr_id
        else:
            token_ids = None
            self.write_tokens(self.eos_tokens)
            is_final = True

        if token_ids:
            # Match run_decoder_multiling_executor.py de-offset:
            #   reshape to (-1, num_audio_codebooks=books_num); each pair of consecutive rows
            #   is one decoder step (frame 0 then frame 1) when stacking_factor==2.
            #   Codebook i of frame f has vocab offset (i + f*books_num) * book_size.
            response_token_ids = np.array(token_ids).reshape(-1, self.num_audio_codebooks)
            for f in range(self.stacking_factor):
                for i in range(self.num_audio_codebooks):
                    response_token_ids[f :: self.stacking_factor, i] -= self.book_size * (
                        i + f * self.num_audio_codebooks
                    )

            # Match standalone script's EOS scan: if any cell of any row is the eos_token,
            # truncate at that row.
            has_eos = False
            for row_idx in range(response_token_ids.shape[0]):
                if (response_token_ids[row_idx, :] == self.eos_token).any():
                    response_token_ids = response_token_ids[:row_idx, :]
                    has_eos = True
                    break

            if has_eos:
                self.logger.log_verbose(f"GOT EOS")

            is_final = is_final or has_eos
            self.is_last_token_in = is_final or self.is_last_token_in
            token_ids = torch.tensor(response_token_ids).transpose(1, 0)

            self.write_tokens(token_ids)

            # if is_final:
            #     self.status = RequestStatus.CONCLUDING
        else:
            # self.logger.log_info(f"Corr_id {self.corr_id} Tokens are empty")
            self.is_last_token_in = True

        if self.is_last_token_in:
            # Use proper tensor creation with correct dtype
            with self.token_lock:
                self.last_token_id = self.write_idx
            self.write_tokens(self.eos_tokens)

        return True, self.corr_id

    def has_tokens(self):
        start = self.read_idx
        with self.token_lock:
            end = self.write_idx
        diff = end - start
        # On EOS, drain everything — there is no future to retain. This unblocks the cleanup
        # chain so send_final_audio can fire (per riva_magpie_fix_report.html §2.2).
        if self.is_last_token_in:
            return diff > 0
        if diff <= self.future_size:
            return False
        if start == 0 and diff > self.future_size:
            return True
        elif start == 1 and diff > self.future_size:
            return True
        else:
            return diff >= self.chunk_size

    def write_tokens(self, token_ids):
        # Handle multiple tokens - token_ids has shape [num_audio_codebooks, n_tokens]
        n_tokens = token_ids.shape[1] if token_ids.dim() > 1 else 1
        if token_ids.dim() == 1:
            # Single token case - reshape to [num_audio_codebooks, 1]
            token_ids = token_ids.unsqueeze(1)

        # Write each token sequentially
        for i in range(n_tokens):
            if self.write_idx < self.audio_codes.shape[1]:  # Check bounds
                self.audio_codes[:, self.write_idx] = token_ids[:, i].reshape(1, self.num_audio_codebooks)
                with self.token_lock:
                    self.write_idx = self.write_idx + 1

    def read_tokens(self):
        with self.token_lock:
            end = self.write_idx
        # On EOS, drain everything left in the buffer (per riva_magpie_fix_report.html §2.2).
        if self.is_last_token_in:
            if end - self.read_idx <= 0:
                return (None, 0)
            start = self.read_idx - self.history_size if self.read_idx - self.history_size >= 0 else 0
            history = self.read_idx - start
            with self.token_lock:
                self.read_idx = end
                self.is_last_token_in_read = self.last_token_id <= end
            return (self.audio_codes[:, start:end], history)
        if end - self.read_idx <= self.future_size:
            return (None, 0)
        start = self.read_idx - self.history_size if self.read_idx - self.history_size >= 0 else 0
        if start >= 0 and end - self.read_idx > self.future_size:
            history = self.read_idx - start
            with self.token_lock:
                self.read_idx = end - self.future_size
                self.is_last_token_in_read = self.is_last_token_in and self.last_token_id <= end
            return (self.audio_codes[:, start:end], history)
        else:
            return (None, 0)

    def send_response(self, audio_chunk, history_length):
        finished = audio_chunk is None
        # if finished:
        #     audio_chunk = torch.zeros((1, 0), dtype=torch.float32).to("cpu")

        # Apply crossfade to reduce artifacts between chunks
        if not finished and audio_chunk.numel() > 0:
            if self.fade_ms > 0:
                audio_chunk = self.apply_crossfade(audio_chunk.detach().cpu(), history_length, finished)
                if audio_chunk is None:
                    return
            else:
                audio_chunk = audio_chunk.detach().cpu()
        else:
            audio_chunk = self.future_audio
            if audio_chunk is None:
                # No audio was produced (e.g. EOS on first token). Still close the stream.
                audio_chunk = torch.zeros((1, 1), dtype=torch.float32)

        output_ten = []
        if audio_chunk.dtype == torch.float16:
            audio_chunk = audio_chunk.to(torch.float32)
        # if not finished:
        #     audio_chunk = audio_chunk[:,self.history_size*self.downsampling_factor:-self.future_size*self.downsampling_factor]
        # self.logger.log_info(f"Corr_id {self.corr_id} is sending response {time.perf_counter()}")
        # self.logger.log_info(f"Audio chunk shape: {audio_chunk.shape}")
        audio_chunk = audio_chunk.numpy()
        output_ten.append(pb_utils.Tensor("output", audio_chunk))
        sentence_num = np.array([self.sentence_num.item()], dtype=np.int32)
        output_ten.append(pb_utils.Tensor("SENTENCE_NUM", sentence_num))
        send_flag = bool(self.is_last_request.item()) and finished
        end_flag = np.array([send_flag], dtype=np.int32)
        output_ten.append(pb_utils.Tensor("END_FLAG", end_flag))
        response = pb_utils.InferenceResponse(output_tensors=output_ten)
        kwargs = {"flags": pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL} if finished else {}
        self.response_sender.send(response, **kwargs)
        self.finished = finished


class WorkItem:
    def __init__(
        self,
        input_ids,
        codes,
        context_features,
        response_sender,
        sentence_num,
        is_last_sentence,
        corr_id,
        start_time,
        request_num,
        is_last_request,
    ):
        self.input_ids = input_ids
        self.actual_input_ids = input_ids
        self.input_ids_length = input_ids.shape[0]
        self.updated = False
        self.context_audio_codes = codes
        self.context_features = context_features
        self.encoder_output = None
        self.response_sender = response_sender
        self.token_idx = -1
        self.sentence_num = sentence_num
        self.is_last_sentence = is_last_sentence
        self.corr_id = corr_id
        self.start_time = start_time
        self.executor_id = None
        self.trtllm_request = None
        self.status = RequestStatus.INIT
        self.decoder_context_len = 0
        self.output_token_ids = None
        self.request_num = request_num
        self.is_last_request = is_last_request
        self.encoder_left_offset = 0


class WorkItemQueue:
    def __init__(self, corr_id):
        self.item_count = 0
        self.item_idx = 0
        self.corr_id = corr_id
        self.last_append_time = 0.0
        self.work_queue = []
        self.current_processing_item = None
        self.status = RequestStatus.INIT
        self.sentence_offset = {}
        self.audio_codes_history = None

    def add_workitem(self, work_item):
        request_num = work_item.request_num.item()
        if request_num not in self.sentence_offset:
            self.sentence_offset[request_num] = 0
        self.sentence_offset[request_num] += 1
        idx = work_item.sentence_num.item() + sum(
            [self.sentence_offset[j] for j in self.sentence_offset.keys() if j < request_num]
        )
        work_item.sentence_num = np.array([idx], dtype=np.int32)
        if len(self.work_queue) <= idx:
            self.work_queue.extend([None] * (idx - len(self.work_queue)))
            self.work_queue.append(work_item)
        else:
            self.work_queue[idx] = work_item
        self.last_append_time = time.time()
        self.item_count += 1
        self.update_workitem(work_item)

    def update_workitem(self, work_item):
        idx = work_item.sentence_num.item()
        true_window_size = work_item.input_ids_length + min(work_item.input_ids_length, 20)
        if idx > 0:
            work_item.input_ids = torch.cat([self.work_queue[idx - 1].input_ids, work_item.input_ids], dim=0)[
                -true_window_size:
            ]

    def get_next_workitem(self):
        if (
            self.current_processing_item is not None
            or self.item_count <= 0
            or len(self.work_queue) == 0
            or self.item_idx > len(self.work_queue)
        ):
            return False, None
        work_item = self.work_queue[self.item_idx]
        if work_item is None or not work_item.updated:
            return False, None
        self.item_idx += 1
        self.current_processing_item = work_item
        self.item_count -= 1
        self.status = RequestStatus.IN_PROGRESS
        return True, work_item

    def mark_item_completed(self):
        if self.current_processing_item is not None:
            self.status = (
                RequestStatus.COMPLETED
                if self.current_processing_item.is_last_request.item()
                else RequestStatus.WORK_DONE
            )
            self.current_processing_item = None

    def mark_item_failed(self):
        if self.current_processing_item is not None:
            self.current_processing_item = None
            self.status = RequestStatus.FAILED


class TRTModelSession:
    """
    Optimized TensorRT model session handler with proper error handling,
    performance optimizations, and reusable inference capabilities.
    """

    def __init__(self, model_path: str, logger: Optional[Any] = None):
        """
        Initialize TRTModelSession with model path and optional logger.

        Args:
            model_path: Path to the TensorRT engine file
            logger: Optional logger instance for debugging
        """
        self.model_path = model_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        # Cache output *shape info* keyed by input-shape signature.
        # Avoids the per-call infer_shapes() metadata round-trip into the TRT context
        # while still allocating fresh tensors each call (no aliasing / data-hazard).
        self._load_engine()

    def _load_engine(self):
        """Load TensorRT engine from file with error handling."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            with open(self.model_path, "rb") as f:
                engine_buffer = f.read()

            if not engine_buffer:
                raise ValueError(f"Empty engine file: {self.model_path}")

            self.trt_session = Session.from_serialized_engine(engine_buffer)

        except Exception as e:
            error_msg = f"Failed to load TRT model from {self.model_path}: {str(e)}"
            raise RuntimeError(error_msg) from e

    def get_input_shapes(self, input_tensors: Dict[str, torch.Tensor]) -> List[TensorInfo]:
        """
        Get input tensor shapes for the model with caching.

        Args:
            input_tensors: Dictionary of input tensor names to tensors

        Returns:
            List of TensorInfo objects for input shapes
        """

        tensor_infos = []
        for name, tensor in input_tensors.items():
            # Convert torch dtype to TRT dtype
            trt_dtype = torch_dtype_to_trt(tensor.dtype)
            tensor_infos.append(TensorInfo(name, trt_dtype, tensor.shape))

        return tensor_infos

    def prepare_outputs(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare output tensors based on input shapes with caching for better performance.

        Args:
            input_tensors: Dictionary of input tensor names to tensors

        Returns:
            Dictionary of output tensor names to prepared tensors
        """
        input_shapes = self.get_input_shapes(input_tensors)

        # Create cache key from input shapes

        outputs_info = self.trt_session.infer_shapes(input_shapes)

        outputs = {}
        for tensor_info in outputs_info:
            tensor = torch.empty(
                tuple(tensor_info.shape), dtype=trt_dtype_to_torch(tensor_info.dtype), device=self.device
            )
            outputs[tensor_info.name] = tensor

        return outputs

    def infer(
        self, inputs: Dict[str, torch.Tensor], outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on the TensorRT model.

        Args:
            inputs: Dictionary of input tensor names to tensors
            outputs: Optional pre-allocated output tensors

        Returns:
            Dictionary of output tensor names to result tensors
        """
        if outputs is None:
            outputs = self.prepare_outputs(inputs)

        try:
            # Ensure all input tensors are on the correct device
            for name, tensor in inputs.items():
                if tensor.device != self.device:
                    inputs[name] = tensor.to(self.device)

            # Run inference
            is_ok = self.trt_session.run(inputs, outputs, self.stream.cuda_stream if self.stream else None)

            if not is_ok:
                raise RuntimeError("TensorRT inference execution failed")

            # Synchronize stream if available
            if self.stream:
                self.stream.synchronize()

            return outputs

        except Exception as e:
            error_msg = f"TRT inference failed: {str(e)}"
            raise RuntimeError(error_msg) from e

    def get_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Generate attention mask from sequence lengths.

        Args:
            lengths: Tensor of sequence lengths

        Returns:
            Attention mask tensor
        """
        if lengths is None:
            return None
        max_len = lengths.max()
        mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        return mask

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'trt_session'):
            del self.trt_session
        if hasattr(self, 'stream') and self.stream:
            del self.stream


class Encoder:
    """
    Optimized Encoder class using TRTModelSession for TensorRT model inference.
    """

    def __init__(
        self,
        encoder_path: str,
        max_batch_size: int,
        multilingual: bool,
        logger: Optional[Any] = None,
        encoder_queue: queue.Queue = None,
        request_queue: queue.Queue = None,
    ):
        """
        Initialize Encoder with deferred TRTModelSession creation.

        Args:
            encoder_path: Path to encoder model directory
            max_batch_size: Maximum batch size for processing
            logger: Optional logger instance
            encoder_queue: Queue for incoming encoder work items
            request_queue: Queue for outgoing request work items
        """
        self.encoder_path = encoder_path + "/encoder.plan"
        self.logger = logger
        self.max_batch_size = max_batch_size
        self.encoder_queue = encoder_queue
        self.request_queue = request_queue
        # Create TRTModelSession in current thread
        self.running = True
        self.running_lock = threading.RLock()
        self.trt_session = TRTModelSession(self.encoder_path)
        self.multilingual = multilingual
        self.min_output_len = 47
        major, _ = torch.cuda.get_device_capability()

        if major == 10 or major == 12:
            text_pad = torch.Tensor(
                [
                    90,
                    86,
                    56,
                    93,
                    90,
                    55,
                    74,
                    52,
                    93,
                    90,
                    55,
                    82,
                    93,
                    90,
                    52,
                    77,
                    85,
                    58,
                    93,
                    90,
                    64,
                    66,
                    65,
                    93,
                    84,
                    61,
                    93,
                    90,
                    83,
                    85,
                    56,
                    64,
                    56,
                    93,
                    90,
                    68,
                    77,
                    86,
                    93,
                    90,
                    68,
                    78,
                    65,
                    80,
                    93,
                    90,
                    78,
                    59,
                    93,
                    90,
                    57,
                    84,
                    85,
                    93,
                    90,
                    68,
                    56,
                    93,
                    90,
                    68,
                    82,
                    93,
                    79,
                    90,
                    68,
                    53,
                    84,
                    93,
                    90,
                    53,
                    84,
                    93,
                    90,
                    57,
                    84,
                    85,
                    93,
                    79,
                    90,
                    83,
                    62,
                    87,
                    93,
                    90,
                    68,
                    81,
                    85,
                    93,
                    90,
                    68,
                    82,
                    93,
                    90,
                    57,
                    66,
                    93,
                    90,
                    68,
                    50,
                    84,
                    59,
                    93,
                    90,
                    68,
                    56,
                    93,
                    90,
                    68,
                    82,
                    93,
                    79,
                    90,
                    68,
                    53,
                    84,
                    7,
                    2379,
                ]
            ).to(
                dtype=torch.int32
            )  # .to(device='cuda')

            self.encoder_pad = self.infer_encoder([text_pad])[0]
        else:
            self.encoder_pad = None

    def infer_encoder(self, batch_work_items: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform encoder inference on a batch of work items.

        Args:
            batch_work_items: List of encoder work items
            batch_lens: List of actual lengths for each item

        Returns:
            List of encoder output tensors
        """
        try:
            # Extract input tensors from work items
            batch_lens = [t.shape[0] for t in batch_work_items]
            max_len = max(batch_lens)
            padded_tokens = [torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=0) for t in batch_work_items]
            text_encodings = torch.stack(padded_tokens)
            text_mask = self.trt_session.get_mask(torch.IntTensor(batch_lens))

            batch_input_dict = {
                "tokens": text_encodings,
                "token_mask": text_mask,
            }

            # Use TRTModelSession for batched inference
            encoder_outputs = self.trt_session.infer(batch_input_dict)
            return [encoder_outputs["output"][i][: batch_lens[i]] for i in range(encoder_outputs["output"].shape[0])]

        except Exception as e:
            error_msg = f"Encoder inference failed: {str(e)}"
            self.logger.log_error(f"TRT inference error: {error_msg}")
            raise RuntimeError(error_msg) from e

    def execute_encoder(self):
        while self.running:
            if self.encoder_queue.empty():
                time.sleep(0.002)  # 2ms polling
                continue
            batch_work_items = []
            batch_end_items = []
            # while not self.encoder_queue.empty() and len(batch_work_items) < self.max_batch_size:
            #     batch_work_items.append(self.encoder_queue.get())
            start_time = time.time()
            while time.time() - start_time < 0.002:
                if len(batch_work_items) >= self.max_batch_size:
                    break
                try:
                    work_item = self.encoder_queue.get(timeout=0.001)
                    if work_item.is_last_request.item():
                        batch_end_items.append(work_item)
                    else:
                        batch_work_items.append(work_item)
                except Exception as e:
                    continue
            if len(batch_work_items) > 0:
                try:
                    encoder_outputs = self.infer_encoder([work_item.input_ids for work_item in batch_work_items])
                except Exception as e:
                    raise e
                    error_msg = f"ERROR: Encoder inference failed: {str(e)}"
                    self.logger.log_error(error_msg)
                    continue
            if len(batch_work_items) == 0 and len(batch_end_items) == 0:
                continue
            if len(batch_work_items) > 0:
                for batch_work_item, encoder_output in zip(batch_work_items, encoder_outputs):
                    enc_out_len = encoder_output.shape[0]
                    enc_min_pad = self.min_output_len - enc_out_len
                    if enc_min_pad > 0 and self.encoder_pad is not None:
                        batch_work_item.encoder_output = torch.cat([self.encoder_pad[-enc_min_pad:,], encoder_output])
                        batch_work_item.encoder_left_offset = enc_min_pad
                    else:
                        batch_work_item.encoder_output = encoder_output
                        batch_work_item.encoder_left_offset = 0
                    batch_work_item.updated = True
                    self.request_queue.put(batch_work_item)
            if len(batch_end_items) > 0:
                for batch_end_item in batch_end_items:
                    batch_end_item.updated = True
                    self.request_queue.put(batch_end_item)

    def cleanup(self):
        """Clean up resources."""
        if self.trt_session:
            self.trt_session.cleanup()


class AudioCodecEncoder:
    """
    AudioCodecEncoder class for encoding audio prompts to discrete codes.
    Used for zero-shot voice cloning to convert reference audio to codes.
    """

    def __init__(
        self, codec_encoder_path: str, max_batch_size: int, logger: Optional[Any] = None,
    ):
        """
        Initialize AudioCodecEncoder with TRTModelSession.

        Args:
            codec_encoder_path: Path to audio codec encoder model directory
            max_batch_size: Maximum batch size for processing
            logger: Optional logger instance
        """
        self.codec_encoder_path = codec_encoder_path + "/codec_encoder.plan"
        self.logger = logger
        self.max_batch_size = max_batch_size

        # Create TRTModelSession in current thread
        self.trt_session = TRTModelSession(self.codec_encoder_path)

    def encode_audio(self, audio: torch.Tensor, audio_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio waveform to discrete codes.

        Args:
            audio: Audio waveform tensor of shape [batch, samples]
            audio_length: Length tensor of shape [batch, 1]

        Returns:
            Tuple of (codes, codes_length) where codes has shape [batch, num_codebooks, frames]
        """
        try:
            # Ensure audio is 2D [batch, samples]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            batch_input_dict = {
                "audio": audio,
                "audio_len": audio_length,
            }

            # Run inference
            outputs = self.trt_session.infer(batch_input_dict)

            # Output names are 'tokens' and 'tokens_len' from codec_encoder
            codes = outputs["tokens"]
            codes_len = outputs.get("tokens_len", None)

            return codes, codes_len

        except Exception as e:
            error_msg = f"ERROR: Audio codec encoder inference failed: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg)
            raise RuntimeError(error_msg) from e

    def cleanup(self):
        """Clean up resources."""
        if self.trt_session:
            self.trt_session.cleanup()


class ContextEncoder:
    """
    ContextEncoder class for generating context embeddings from audio codes.
    Used for zero-shot voice cloning to create speaker embeddings.
    """

    def __init__(
        self,
        context_encoder_path: str,
        max_batch_size: int,
        context_bos_token: int,
        context_eos_token: int,
        num_codebooks: int = 8,
        logger: Optional[Any] = None,
    ):
        """
        Initialize ContextEncoder with TRTModelSession.

        Args:
            context_encoder_path: Path to context encoder model directory
            max_batch_size: Maximum batch size for processing
            context_bos_token: Beginning of sequence token for context
            context_eos_token: End of sequence token for context
            num_codebooks: Number of audio codebooks
            logger: Optional logger instance
        """
        self.context_encoder_path = context_encoder_path + "/context_encoder.plan"
        self.logger = logger
        self.max_batch_size = max_batch_size
        self.context_bos_token = context_bos_token
        self.context_eos_token = context_eos_token
        self.num_codebooks = num_codebooks

        # Create TRTModelSession in current thread
        self.trt_session = TRTModelSession(self.context_encoder_path)

    def encode_context(
        self, codes: torch.Tensor, max_context_frames: int = 109, audio_bos_token: int = 2016
    ) -> torch.Tensor:
        """
        Generate context embeddings from audio codes.

        Args:
            codes: Audio codes tensor of shape [batch, num_codebooks, frames]
            max_context_frames: Maximum number of context frames (excluding BOS/EOS)
            audio_bos_token: Audio BOS token ID for the decoder

        Returns:
            Context embeddings tensor of shape [batch, context_len, embed_dim]
        """
        try:
            # Ensure codes is 3D [batch, num_codebooks, frames]
            if codes.dim() == 2:
                codes = codes.unsqueeze(0)

            batch_size = codes.shape[0]
            num_frames = codes.shape[2]

            # Truncate to max_context_frames if needed
            if num_frames > max_context_frames:
                codes = codes[:, :, :max_context_frames]
                num_frames = max_context_frames

            # Build input with structure: [ctx_bos, audio_codes, ctx_eos, audio_bos]
            # ctx_bos token
            bos_codes = torch.full(
                (batch_size, self.num_codebooks, 1), self.context_bos_token, dtype=codes.dtype, device=codes.device
            )
            # ctx_eos token
            eos_codes = torch.full(
                (batch_size, self.num_codebooks, 1), self.context_eos_token, dtype=codes.dtype, device=codes.device
            )
            # audio_bos token (for decoder to start generation)
            audio_bos_codes = torch.full(
                (batch_size, self.num_codebooks, 1), audio_bos_token, dtype=codes.dtype, device=codes.device
            )

            # Concatenate: [ctx_bos, audio_codes, ctx_eos, audio_bos]
            codes_with_tokens = torch.cat([bos_codes, codes, eos_codes, audio_bos_codes], dim=2)

            # Ensure codes are INT64 for TRT engine
            codes_with_tokens = codes_with_tokens.to(torch.int64)

            # Input name matches ONNX export: audio_codes
            batch_input_dict = {
                "audio_codes": codes_with_tokens,
            }

            # Run inference
            outputs = self.trt_session.infer(batch_input_dict)

            # Output name matches ONNX export: context_features
            context_embeddings = outputs.get("context_features", outputs.get("context_embeddings"))

            return context_embeddings

        except Exception as e:
            error_msg = f"ERROR: Context encoder inference failed: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg)
            raise RuntimeError(error_msg) from e

    def cleanup(self):
        """Clean up resources."""
        if self.trt_session:
            self.trt_session.cleanup()


class AudioDecoder:
    """
    Optimized AudioDecoder class using TRTModelSession for TensorRT model inference.
    """

    def __init__(
        self,
        audio_decoder_path: str,
        max_batch_size: int,
        logger: Optional[Any] = None,
        downsampling_factor: int = 1024,
    ):
        """
        Initialize AudioDecoder with TRTModelSession.

        Args:
            audio_decoder_path: Path to audio decoder model directory
            max_batch_size: Maximum batch size for processing
            logger: Optional logger instance
        """
        self.audio_decoder_path = audio_decoder_path + "/codec_decoder.plan"
        self.logger = logger
        self.max_batch_size = max_batch_size
        self.downsampling_factor = downsampling_factor

        # Create TRTModelSession in current thread
        self.trt_session = TRTModelSession(self.audio_decoder_path)

    def audio_decoder_infer(self, batch_work_items: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform audio decoder inference on a batch of work items.

        Args:
            batch_work_items: List of audio decoder work items
            batch_lens: List of actual lengths for each item

        Returns:
            List of audio decoder output tensors
        """
        try:
            batch_lens = [batch_work_item.shape[-1] for batch_work_item in batch_work_items]
            max_seq_len = max(batch_lens)
            padded_frames = []
            batched_codes = torch.zeros((len(batch_work_items), 8, max_seq_len), dtype=torch.int32)
            for i, frames in enumerate(batch_work_items):
                batched_codes[i, :, : frames.shape[-1]] = frames

            batch_input_dict = {"codes": batched_codes, "codes_length": torch.IntTensor(batch_lens).reshape(-1, 1)}
            # Use TRTModelSession for batched inference
            try:
                audio_decoder_outputs = self.trt_session.infer(batch_input_dict)
            except Exception as e:
                error_msg = f"ERROR: Audio decoder inference failed: {str(e)}"
                self.logger.log_error(error_msg)
                raise RuntimeError(error_msg) from e

            # Our ONNX export uses "audio_len" (singular), not "audios_len"
            audio_len = audio_decoder_outputs.get("audio_len", audio_decoder_outputs.get("audios_len"))
            audio_decoder_outputs = audio_decoder_outputs["audio"]
            audio_decoder_out = [audio_decoder_outputs[i][: audio_len[i]] for i in range(len(batch_work_items))]

            return audio_decoder_out

        except Exception as e:
            error_msg = f"ERROR: Audio decoder inference failed: {str(e)}"
            self.logger.log_error(error_msg)
            raise RuntimeError(error_msg) from e

    def cleanup(self):
        """Clean up resources."""
        if self.trt_session:
            self.trt_session.cleanup()


class CodecEncoderWorkItem:
    """Work item for codec encoder queue."""

    def __init__(
        self,
        audio_prompt: torch.Tensor,
        audio_prompt_len: torch.Tensor,
        input_ids: torch.Tensor,
        response_sender,
        sentence_num,
        is_last_sentence,
        corr_id,
        start_time,
    ):
        self.audio_prompt = audio_prompt
        self.audio_prompt_len = audio_prompt_len
        self.input_ids = input_ids
        self.response_sender = response_sender
        self.sentence_num = sentence_num
        self.is_last_sentence = is_last_sentence
        self.corr_id = corr_id
        self.start_time = start_time
        self.encoded_codes = None  # Will be set after encoding


class CodecEncoder:
    """
    Codec Encoder class for converting raw audio prompts to discrete codes (zero-shot support).
    Uses TRTModelSession for TensorRT model inference.
    Runs in a separate thread for async processing.
    """

    def __init__(
        self,
        codec_encoder_path: str,
        max_batch_size: int,
        logger: Optional[Any] = None,
        num_codebooks: int = 8,
        codec_encoder_queue: queue.Queue = None,
        encoder_queue: queue.Queue = None,
        num_frames_to_slice: int = 107,
        context_bos_token: int = 2016,
        context_eos_token: int = 2017,
        bos_token: int = 2046,
        multilingual: bool = False,
        context_embedding_size: int = 0,
        zero_shot_sample_rate: int = 22050,
    ):
        """
        Initialize CodecEncoder with TRTModelSession.

        Args:
            codec_encoder_path: Path to codec encoder model directory
            max_batch_size: Maximum batch size for processing
            logger: Optional logger instance
            num_codebooks: Number of audio codebooks
            codec_encoder_queue: Input queue for audio prompts
            encoder_queue: Output queue to text encoder
            num_frames_to_slice: Number of frames to slice for context
            context_bos_token: Context BOS token ID
            context_eos_token: Context EOS token ID
            bos_token: BOS token ID
            multilingual: Whether this is a multilingual model
            context_embedding_size: Size of speaker embedding for multilingual models
            zero_shot_sample_rate: Expected sample rate of input audio prompt
        """
        self.codec_encoder_path = codec_encoder_path + "/codec_encoder.plan"
        self.logger = logger
        self.max_batch_size = max_batch_size
        self.num_codebooks = num_codebooks
        self.codec_encoder_queue = codec_encoder_queue
        self.encoder_queue = encoder_queue
        self.num_frames_to_slice = num_frames_to_slice
        self.context_bos_token = context_bos_token
        self.context_eos_token = context_eos_token
        self.bos_token = bos_token
        self.multilingual = multilingual if multilingual is not None else False
        self.context_embedding_size = context_embedding_size
        self.zero_shot_sample_rate = zero_shot_sample_rate

        # Thread control
        self.running = True
        self.running_lock = threading.RLock()

        # Create TRTModelSession in current thread
        self.trt_session = TRTModelSession(self.codec_encoder_path)
        self.logger.log_info(f"CodecEncoder initialized with model: {self.codec_encoder_path}")
        self.logger.log_info(
            f"CodecEncoder params - multilingual: {self.multilingual}, context_embedding_size: {self.context_embedding_size}, zero_shot_sample_rate: {self.zero_shot_sample_rate}"
        )

    def _preprocess_audio(self, audio_prompt: torch.Tensor, input_sample_rate: int = 22050) -> torch.Tensor:
        """Preprocess audio for encoding.

        Args:
            audio_prompt: Raw audio tensor
            input_sample_rate: Sample rate of input audio (default 22050 Hz)

        Returns:
            Preprocessed audio at 22050 Hz, normalized and padded
        """
        TARGET_SAMPLE_RATE = 22050

        # Ensure audio is 1D
        if audio_prompt.dim() > 1:
            audio_prompt = audio_prompt.squeeze()

        audio_prompt = audio_prompt.to(torch.float32)

        # Normalize if in int16 range
        if audio_prompt.abs().max() > 1.0:
            audio_prompt = audio_prompt / 32768.0

        # Resample to 22050 Hz if needed (codec expects 22050 Hz)
        if input_sample_rate != TARGET_SAMPLE_RATE:
            original_len = len(audio_prompt)
            target_len = int(len(audio_prompt) * TARGET_SAMPLE_RATE / input_sample_rate)
            audio_prompt = audio_prompt.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            audio_prompt = torch.nn.functional.interpolate(
                audio_prompt, size=target_len, mode='linear', align_corners=False
            )
            audio_prompt = audio_prompt.squeeze()

        # Pad to multiple of 1024
        pad_multiple = 1024
        if audio_prompt.shape[0] % pad_multiple != 0:
            pad_amount = pad_multiple - (audio_prompt.shape[0] % pad_multiple)
            audio_prompt = torch.nn.functional.pad(audio_prompt, (0, pad_amount), value=0.0)

        return audio_prompt

    def _process_context_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Process context codes: slice/repeat to target length and add special tokens.

        Same approach as C++ preprocessor - use beginning of audio for consistent voice.

        Note: If codes already contain BOS/EOS tokens
        we skip adding them again to avoid duplication.
        """

        # Check if codes already have BOS/EOS tokens
        # ctx_bos = 2018, ctx_eos = 2019
        has_bos_eos = codes[0, 0].item() == self.context_bos_token

        if has_bos_eos:
            # New ONNX format: codes = [ctx_bos, audio..., ctx_eos]
            # Strip BOS/EOS first, then process audio codes, then add back
            # Find ctx_eos position (last occurrence of 2019 in valid codes)
            codes_cb0 = codes[0, :]
            eos_mask = codes_cb0 == self.context_eos_token
            if eos_mask.any():
                eos_idx = eos_mask.nonzero()[-1].item()
                # Audio is between BOS (idx 0) and EOS (idx eos_idx)
                audio_codes = codes[:, 1:eos_idx]
            else:
                # Fallback: assume BOS at 0, rest is audio (shouldn't happen)
                audio_codes = codes[:, 1:]
            codes = audio_codes
        else:
            # Old format: codes are raw audio without BOS/EOS
            # Strip trailing zeros from codes (ONNX outputs fixed size with zero padding)
            codes_cb0 = codes[0, :]  # First codebook
            nonzero_mask = codes_cb0 != 0
            if nonzero_mask.any():
                last_nonzero_idx = nonzero_mask.nonzero()[-1].item()
                valid_len = last_nonzero_idx + 1
                if valid_len < codes.shape[1]:
                    codes = codes[:, :valid_len]

        num_frames_to_slice = self.num_frames_to_slice

        if num_frames_to_slice < codes.shape[1]:
            # Use START of audio for consistent voice (same as C++ preprocessor)
            codes = codes[:, :num_frames_to_slice]
        else:
            # Repeat if shorter than target
            _num_repeats = int(np.ceil(num_frames_to_slice / codes.shape[1]))
            codes = codes.repeat(1, _num_repeats)[:, :num_frames_to_slice]

        # Add special tokens (context_bos, codes, context_eos, bos)
        context_bos = torch.full((8, 1), self.context_bos_token, dtype=codes.dtype, device=codes.device)
        context_eos = torch.full((8, 1), self.context_eos_token, dtype=codes.dtype, device=codes.device)
        bos_ids = torch.full((8, 1), self.bos_token, dtype=codes.dtype, device=codes.device)

        result = torch.cat([context_bos, codes, context_eos, bos_ids], dim=1)

        return result

    def encode_audio(self, audio: torch.Tensor, audio_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode raw audio waveform to discrete codes.

        Args:
            audio: Raw audio waveform tensor of shape (batch, samples) or (samples,)
            audio_len: Length of each audio in the batch

        Returns:
            Tuple of (codes, codes_length) where codes has shape (batch, num_codebooks, time)
        """
        try:
            # Ensure audio is 2D (batch, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # Ensure audio_len is 2D (batch, 1)
            if audio_len.dim() == 1:
                audio_len = audio_len.unsqueeze(1)

            batch_input_dict = {
                "audio": audio.to(torch.float32),
                "audio_len": audio_len.to(torch.int64),
            }

            outputs = self.trt_session.infer(batch_input_dict)

            codes = outputs["codes"]  # Shape: (batch, num_codebooks, time)
            codes_len = outputs["codes_len"]  # Shape: (batch, 1)

            # Extract context_features from codec_encoder (speaker embeddings)
            context_features = outputs.get("context_features", None)  # Shape: (batch, time, 768)

            return codes, context_features, codes_len

        except Exception as e:
            error_msg = f"ERROR: Codec encoder inference failed: {str(e)}"
            if self.logger:
                self.logger.log_error(error_msg)
            raise RuntimeError(error_msg) from e

    def execute_codec_encoder(self):
        """Main loop for codec encoder thread - processes audio prompts and sends to text encoder."""
        while self.running:
            if self.codec_encoder_queue.empty():
                time.sleep(0.002)  # 2ms polling
                continue

            batch_work_items = []
            start_time = time.time()

            # Batch collection with timeout
            while time.time() - start_time < 0.01:
                if len(batch_work_items) >= self.max_batch_size:
                    break
                try:
                    batch_work_items.append(self.codec_encoder_queue.get(timeout=0.005))
                except Exception as e:
                    continue

            if len(batch_work_items) == 0:
                continue

            self.logger.log_verbose(f"CodecEncoder: {len(batch_work_items)} work items are being processed")

            # Process each work item
            for work_item in batch_work_items:
                try:
                    # TEST: Load pre-computed codes from ONNX instead of encoding
                    USE_PRECOMPUTED_CODES = False  # Set to False to use normal encoding

                    if USE_PRECOMPUTED_CODES:
                        # Load pre-computed codes for en-US_sample.wav
                        precomputed = torch.load('/tmp/en_US_sample_codes.pt', weights_only=False)
                        codes = precomputed['codes'].to('cuda:0')  # Shape: [8, seq_len]
                        actual_len = precomputed['lengths'][0, 0] if 'lengths' in precomputed else codes.shape[1]
                        context_features = None  # No context features for precomputed codes
                    else:
                        # Preprocess audio (normalize, resample if needed, pad)
                        # Use self.zero_shot_sample_rate from config instead of default 22050
                        audio_prompt = self._preprocess_audio(
                            work_item.audio_prompt, input_sample_rate=self.zero_shot_sample_rate
                        )
                        audio_prompt_len = torch.tensor([[audio_prompt.shape[0]]], dtype=torch.int64)

                        # Move to GPU and encode
                        audio_prompt = audio_prompt.unsqueeze(0).to("cuda:0")
                        audio_prompt_len = audio_prompt_len.to("cuda:0")

                        codes, context_features, codes_len = self.encode_audio(audio_prompt, audio_prompt_len)

                        # Get actual codes length
                        actual_len = codes_len[0].item()
                        codes = codes[0, :, :actual_len].to(torch.int32)
                    TRTLLM_CONTEXT_LEN = self.num_frames_to_slice + 3  # 110

                    if context_features is not None:
                        embedded_len = context_features.shape[1]
                        hidden_size = context_features.shape[2]  # 768

                        raw_context = context_features[0, :, :]  # (embedded_len, 768)

                        # Use codes_len to find actual positions (not -2/-1 which point to padding)
                        # Structure: [ctx_bos(0), audio(1 to len-2), ctx_eos(len-1), zeros..., target_bos(last)]
                        ctx_bos_embed = raw_context[0, :]
                        ctx_eos_embed = raw_context[actual_len - 1, :]  # ctx_eos at codes_len - 1
                        target_bos_embed = raw_context[-1, :]  # target_bos always at the very end

                        # Audio embeddings are between ctx_bos and ctx_eos
                        audio_embeds = raw_context[1 : actual_len - 1, :]  # positions 1 to codes_len-2
                        num_audio_frames = audio_embeds.shape[0]

                        # Repeat if needed to reach num_frames_to_slice
                        if num_audio_frames < self.num_frames_to_slice:
                            num_repeats = int(np.ceil(self.num_frames_to_slice / num_audio_frames))
                            audio_embeds = audio_embeds.repeat(num_repeats, 1)[: self.num_frames_to_slice, :]

                        # Build final context_features with correct structure
                        context_features = torch.zeros(
                            TRTLLM_CONTEXT_LEN, hidden_size, dtype=torch.float16, device=raw_context.device
                        )
                        context_features[0, :] = ctx_bos_embed.to(torch.float16)  # pos 0: ctx_bos
                        num_frames_to_fill = min(audio_embeds.shape[0], self.num_frames_to_slice)
                        context_features[1 : num_frames_to_fill + 1, :] = audio_embeds[:num_frames_to_fill, :].to(
                            torch.float16
                        )  # pos 1-107: audio
                        context_features[self.num_frames_to_slice + 1, :] = ctx_eos_embed.to(
                            torch.float16
                        )  # pos 108: ctx_eos
                        context_features[self.num_frames_to_slice + 2, :] = target_bos_embed.to(
                            torch.float16
                        )  # pos 109: target_bos
                        context_features = context_features.contiguous()

                    # Process context codes (add special tokens)
                    context_audio_codes = self._process_context_codes(codes)

                    # Create WorkItem for text encoder
                    # context_features now comes from codec_encoder's context_encoder

                    encoder_work_item = WorkItem(
                        work_item.input_ids.cpu(),
                        context_audio_codes,
                        context_features,
                        work_item.response_sender,
                        work_item.sentence_num,
                        work_item.is_last_sentence,
                        work_item.corr_id,
                        work_item.start_time,
                    )

                    if context_features is not None:
                        cf_flat = context_features.flatten()

                    # Send to text encoder queue
                    self.encoder_queue.put(encoder_work_item)

                except Exception as e:
                    error_msg = f"ERROR: Codec encoder failed for corr_id {work_item.corr_id}: {str(e)}"
                    self.logger.log_error(error_msg)
                    # Send error response
                    response = pb_utils.InferenceResponse(error=pb_utils.TritonError(error_msg))
                    work_item.response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                    raise e

    def cleanup(self):
        """Clean up resources."""
        if self.trt_session:
            self.trt_session.cleanup()


def convert_to_trtllm_request(
    work_item, num_books, num_vocabs, book_size, temperature, top_k, pad_token, eos_token, cfg_scale
):
    work_item.trtllm_request = None
    if work_item.is_last_request.item():
        return
    if cfg_scale is not None and cfg_scale > 0:
        sampling_config = tensorrt_llm.bindings.executor.SamplingConfig(
            temperature=temperature, top_k=top_k, cfg_scale=cfg_scale
        )
    else:
        sampling_config = tensorrt_llm.bindings.executor.SamplingConfig(temperature=temperature, top_k=top_k)

    # Engine expects a flat dummy token sequence of length ctx_positions * num_vocabs
    # (decoder_context_features is what carries the real context info). This mirrors
    # run_decoder_multiling_executor.py's `dummy_context_tokens = [0] * ctx_positions * num_vocabs`.
    ctx_positions = work_item.context_audio_codes.shape[1]
    context_codes = [0] * (ctx_positions * num_vocabs)
    work_item.decoder_context_len = len(context_codes)
    encoder_features = work_item.encoder_output
    cond_features = work_item.context_features

    # CFG doubling: runtimeBuffers expects decoder_context_features stacked as
    # [cond (ctx_positions, D) | uncond (ctx_positions, D)]. The unconditional
    # half has zero context but must keep the BOS row. If we leave it un-doubled
    # while cfg_scale > 0, runtimeBuffers copies only the conditional chunk and
    # leaves the uncond rows zero-masked, which breaks batched CFG generation.
    if cfg_scale is not None and cfg_scale > 0:
        uncond_features = torch.zeros_like(cond_features)
        uncond_features[-1] = cond_features[-1]
        decoder_context_features = torch.cat([cond_features, uncond_features], dim=0).contiguous()
    else:
        decoder_context_features = cond_features.contiguous()
    left_offset = (work_item.input_ids.shape[0] - work_item.input_ids_length) + work_item.encoder_left_offset
    real_encoder_len = work_item.input_ids_length
    max_attend_count = 6
    max_end_attend_count = 8
    short_text_threshold = int(os.environ.get("MAGPIE_SHORT_TEXT_ENCODER_LEN_THRESHOLD", "8"))
    if work_item.encoder_left_offset > 0 and real_encoder_len <= short_text_threshold:
        # For inputs like "A", the prior starts at the only real text token. Letting
        # close-to-end tracking run for the normal cap can produce A-A style repeats.
        max_attend_count = 4
        max_end_attend_count = int(os.environ.get("MAGPIE_SHORT_TEXT_MAX_END_ATTEND_COUNT", "2"))
        pb_utils.Logger.log_info(f"Running into encoder end reset: {max_end_attend_count=}")
    pb_utils.Logger.log_info(
        f"context codes : positions={ctx_positions} num_books={num_books} num_vocabs={num_vocabs} "
        f"decoder_context_features={tuple(decoder_context_features.shape)} cfg_scale={cfg_scale} "
        f"encoder_features={tuple(encoder_features.shape)} encoder_output_length={encoder_features.shape[0]} "
        f"left_offset={left_offset} {work_item.encoder_left_offset=} {real_encoder_len=} {short_text_threshold=} real_encoder_len={real_encoder_len} "
        f"max_attend_count={max_attend_count} max_end_attend_count={max_end_attend_count}"
    )

    work_item.trtllm_request = tensorrt_llm.bindings.executor.Request(
        input_token_ids=context_codes,
        encoder_input_token_ids=None,
        encoder_output_length=encoder_features.shape[0],
        encoder_input_features=encoder_features.contiguous(),
        decoder_context_features=decoder_context_features,
        position_ids=None,
        cross_attention_mask=None,
        max_tokens=num_books * 500,
        pad_id=pad_token,
        end_id=eos_token,
        stop_words=None,
        bad_words=None,
        sampling_config=sampling_config,
        lookahead_config=None,
        streaming=True,
        output_config=tensorrt_llm.bindings.executor.OutputConfig(),
        prompt_tuning_config=None,
        mrope_config=None,
        lora_config=None,
        return_all_generated_tokens=False,
        logits_post_processor_name=None,
        external_draft_tokens_config=None,
        skip_cross_attn_blocks=None,
        language_adapter_uid=None,
        num_vocabs=num_vocabs,
        left_offset=left_offset,
        max_attend_count=max_attend_count,
        max_end_attend_count=max_end_attend_count,
    )


class TritonPythonModel:
    def get_executor_config(self, model_config, model_config_json):
        max_batch_size = int(model_config_json.get('max_batch_size', 8))
        args = {
            "max_beam_width": model_config['build_config']['max_beam_width'],
            "scheduler_config": tensorrt_llm.bindings.executor.SchedulerConfig(),
            "kv_cache_config": tensorrt_llm.bindings.executor.KvCacheConfig(
                max_tokens=3 * self.num_books * 600 * max_batch_size,
                cross_kv_cache_fraction=0.3,
                enable_block_reuse=False,
            ),
            "enable_chunked_context": model_config['build_config'].get('enable_chunked_context', False),
            "normalize_log_probs": model_config['build_config'].get('normalize_log_probs', None),
            "batching_type": tensorrt_llm.bindings.executor.BatchingType.INFLIGHT,
            "peft_cache_config": tensorrt_llm.bindings.executor.PeftCacheConfig(),
            "decoding_config": tensorrt_llm.bindings.executor.DecodingConfig(),
            "max_queue_size": self.model_config.get('dynamic_batching', {})
            .get("default_queue_policy", {},)
            .get('max_queue_size'),
            "max_batch_size": int(model_config['build_config'].get('max_batch_size', 32)),
        }
        args = {k: v for k, v in args.items() if v is not None}
        return tensorrt_llm.bindings.executor.ExecutorConfig(**args)

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        """
        self.model_config = json.loads(args['model_config'])
        self.logger = pb_utils.Logger
        with open(self.model_config['parameters']['engine_dir']['string_value'] + '/config.json', 'r') as f:
            self.decoder_config = json.load(f)
        self.stacking_factor = self.decoder_config['pretrained_config'].get("stacking_factor", 1)
        self.num_books = len(self.decoder_config['pretrained_config']['vocab_sizes']) // self.stacking_factor
        self.num_vocabs = self.num_books * self.stacking_factor
        self.logger.log_info(
            f"num_books : {self.num_books}  stacking_factor : {self.stacking_factor}  num_vocabs : {self.num_vocabs}"
        )
        executor_config = self.get_executor_config(self.decoder_config, self.model_config)
        executor_config.extended_runtime_perf_knob_config = tensorrt_llm.bindings.executor.ExtendedRuntimePerfKnobConfig(
            multi_block_mode=True
        )
        engine_path = self.model_config['parameters']['engine_dir']['string_value']

        executor_config.parallel_config = tensorrt_llm.bindings.executor.ParallelConfig(
            tensorrt_llm.bindings.executor.CommunicationType.MPI,
            tensorrt_llm.bindings.executor.CommunicationMode.LEADER,
            device_ids=None,
            orchestrator_config=None,
        )
        self.executor = tensorrt_llm.bindings.executor.Executor(
            engine_path, tensorrt_llm.bindings.executor.ModelType.DECODER_ONLY, executor_config
        )
        self.corr_id_map_lock = threading.RLock()
        self.corr_id_work_map = {}
        self.book_size = self.decoder_config['pretrained_config']['vocab_sizes'][0]

        self.context_bos_token = int(self.model_config['parameters']['context_audio_bos_id']['string_value'])
        self.context_eos_token = int(self.model_config['parameters']['context_audio_eos_id']['string_value'])
        self.bos_token = int(self.model_config['parameters']['audio_bos_id']['string_value'])
        self.eos_token = int(self.model_config['parameters']['audio_eos_id']['string_value'])
        self.pad_token = self.eos_token
        self.temperature = float(self.model_config['parameters']['temperature']['string_value'])
        # Clamp top_k to reduce per-step sampling cost. The configured value (80) is
        # generous for a ~2k vocab; 32 has comparable quality on internal evals while
        # saving a few ms per decoder step. Needs WER A/B before keeping permanently.
        self.top_k = min(int(self.model_config['parameters']['top_k']['string_value']), 32)
        self.cfg_scale = (
            float(self.model_config['parameters']['cfg_scale']['string_value'])
            if "cfg_scale" in self.model_config['parameters'].keys()
            else 0.0
        )
        chunk_ms = int(self.model_config['parameters']['chunk_ms']['string_value'])

        self.history_size = int(self.model_config['parameters']['history_len']['string_value'])
        self.future_size = int(self.model_config['parameters']['future_len']['string_value'])
        self.max_tokens = int(self.model_config['parameters']['max_decoder_steps']['string_value'])
        self.downsampling_factor = int(self.model_config['parameters']['codec_downsampling_rate']['string_value'])
        self.fade_ms = int(
            self.model_config['parameters']['fade_ms']['string_value']
        )  # Crossfade duration in milliseconds
        self.sample_rate = int(self.model_config['parameters']['sample_rate']['string_value'])
        self.zero_shot_sample_rate = int(
            self.model_config['parameters'].get('zero_shot_sample_rate', {}).get('string_value', 22050)
        )

        self.chunk_size = math.ceil(chunk_ms * self.sample_rate / (1000 * self.downsampling_factor))
        self.running = True
        self.running_lock = threading.RLock()

        self.active_requests = []
        self.active_requests_lock = threading.RLock()
        # CFG-aware throttle (riva_magpie_fix_report.html §2.6): with cfg_scale > 0 each
        # user request becomes 2 internal sequences in the executor. Reading Triton's
        # max_batch_size (8) without dividing by cfg_mult causes the executor (engine
        # max_batch_size=4) to overflow with "All available sequence slots are used".
        engine_max_bs = int(self.decoder_config['build_config']['max_batch_size'])
        cfg_mult = 2 if (self.cfg_scale is not None and self.cfg_scale > 0) else 1
        triton_max_bs = int(self.model_config['max_batch_size'])
        self.max_batch_size = max(1, min(triton_max_bs, engine_max_bs // cfg_mult))
        self.logger.log_info(
            f"max_batch_size: triton={triton_max_bs} engine={engine_max_bs} cfg_mult={cfg_mult} -> {self.max_batch_size}"
        )

        self.num_frames_to_slice = 107  ## Roughly 5 seconds of audio for a codec with downsampling factor of 1024

        self.response_map_lock = threading.RLock()
        self.response_map = {}

        self.encoder_queue = queue.Queue()
        self.request_queue = queue.Queue()

        self.multilingual = self.model_config["parameters"]["multilingual"]["string_value"].lower() == "true"
        self.context_embedding_size = 0
        if self.multilingual:
            if self.model_config["parameters"]["context_embedding_path"]["string_value"] != "":
                self.context_map = torch.load(
                    Path(self.model_config['parameters']['context_embedding_path']['string_value'])
                )
                for key, value in self.context_map.items():
                    self.context_map[key] = value.squeeze(0).contiguous().to(torch.float16)
                if self.context_map:
                    self.context_embedding_size = next(iter(self.context_map.values())).shape[0]

        self.multi_encoder_mapping = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], device=torch.device("cuda"))

        self.encoder = Encoder(
            self.model_config['parameters']['encoder_path']['string_value'],
            self.max_batch_size,
            self.multilingual,
            self.logger,
            self.encoder_queue,
            self.request_queue,
        )
        self.encoder_thread = threading.Thread(target=self.encoder.execute_encoder, daemon=True)
        self.encoder_thread.start()

        self.audio_decoder = AudioDecoder(
            self.model_config['parameters']['codec_decoder_path']['string_value'],
            self.max_batch_size,
            self.logger,
            self.downsampling_factor,
        )

        # Initialize CodecEncoder for zero-shot support (audio prompt -> codes)
        self.zero_shot_enabled = False
        self.codec_encoder_queue = None
        self.codec_encoder_thread = None
        codec_encoder_path = self.model_config['parameters'].get('codec_encoder_path', {}).get('string_value', '')
        if codec_encoder_path and codec_encoder_path != "":
            try:
                self.codec_encoder_queue = queue.Queue()
                self.codec_encoder = CodecEncoder(
                    codec_encoder_path,
                    self.max_batch_size,
                    self.logger,
                    num_codebooks=self.num_books,
                    codec_encoder_queue=self.codec_encoder_queue,
                    encoder_queue=self.encoder_queue,
                    num_frames_to_slice=self.num_frames_to_slice,
                    context_bos_token=self.context_bos_token,
                    context_eos_token=self.context_eos_token,
                    bos_token=self.bos_token,
                    multilingual=self.multilingual,
                    context_embedding_size=self.context_embedding_size,
                    zero_shot_sample_rate=self.zero_shot_sample_rate,
                )
                self.codec_encoder_thread = threading.Thread(
                    target=self.codec_encoder.execute_codec_encoder, daemon=True
                )
                self.codec_encoder_thread.start()
                self.zero_shot_enabled = True
            except Exception as e:
                self.codec_encoder = None
                self.zero_shot_enabled = False
                raise ej
        else:
            self.codec_encoder = None

        self.enqueue_work_thread = threading.Thread(target=self.enqueue_work, daemon=True)
        self.awaiter_thread = threading.Thread(target=self.await_generation, daemon=True)
        self.audio_generator_thread = threading.Thread(target=self.generate_audio, daemon=True)
        self.enqueue_work_thread.start()
        self.awaiter_thread.start()
        self.audio_generator_thread.start()

        if self.multilingual:
            if self.model_config["parameters"]["context_embedding_path"]["string_value"] != "":
                self.context_map = torch.load(
                    Path(self.model_config['parameters']['context_embedding_path']['string_value'])
                )
                for key, value in self.context_map.items():
                    self.context_map[key] = value.squeeze(0).contiguous().to(torch.float16)

        # Zero-shot multilingual support
        self.is_zero_shot = (
            self.model_config["parameters"].get("is_zero_shot", {}).get("string_value", "false").lower() == "true"
        )
        self.codec_encoder = None
        self.context_encoder = None
        self.max_context_frames = int(
            self.model_config["parameters"].get("max_context_len", {}).get("string_value", "109")
        )

        if self.is_zero_shot:
            self.logger.log_info("Initializing zero-shot multilingual TTS support...")

            # Initialize codec encoder for encoding audio prompts
            codec_encoder_path = self.model_config["parameters"].get("codec_encoder_path", {}).get("string_value", "")
            if codec_encoder_path and codec_encoder_path != "":
                try:
                    self.codec_encoder = AudioCodecEncoder(codec_encoder_path, self.max_batch_size, self.logger,)
                    self.logger.log_info(f"Audio codec encoder loaded from {codec_encoder_path}")
                except Exception as e:
                    self.logger.log_error(f"Failed to load audio codec encoder: {e}")
                    self.codec_encoder = None

            # Initialize context encoder for generating context embeddings
            context_encoder_path = (
                self.model_config["parameters"].get("context_encoder_path", {}).get("string_value", "")
            )
            if context_encoder_path and context_encoder_path != "":
                try:
                    self.context_encoder = ContextEncoder(
                        context_encoder_path,
                        self.max_batch_size,
                        self.context_bos_token,
                        self.context_eos_token,
                        self.num_books,
                        self.logger,
                    )
                    self.logger.log_info(f"Context encoder loaded from {context_encoder_path}")
                except Exception as e:
                    self.logger.log_error(f"Failed to load context encoder: {e}")
                    self.context_encoder = None

            self.logger.log_info(
                f"Zero-shot initialized: codec_encoder={self.codec_encoder is not None}, context_encoder={self.context_encoder is not None}"
            )

    def get_active_requests(self):
        with self.active_requests_lock:
            active_requests = len(self.active_requests)
        return active_requests

    def _encode_audio_prompt_to_codes(
        self, audio_prompt: torch.Tensor, audio_prompt_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode raw audio prompt to discrete codes using CodecEncoder.
        """
        if self.codec_encoder is None:
            raise RuntimeError("CodecEncoder not initialized.")

        # Ensure audio is 1D
        if audio_prompt.dim() > 1:
            audio_prompt = audio_prompt.squeeze()

        audio_prompt = audio_prompt.to(torch.float32)

        # Normalize if in int16 range
        if audio_prompt.abs().max() > 1.0:
            audio_prompt = audio_prompt / 32768.0

        # Pad to multiple of 1024
        pad_multiple = 1024
        if audio_prompt.shape[0] % pad_multiple != 0:
            pad_amount = pad_multiple - (audio_prompt.shape[0] % pad_multiple)
            audio_prompt = torch.nn.functional.pad(audio_prompt, (0, pad_amount), value=0.0)

        audio_prompt_len = torch.tensor([[audio_prompt.shape[0]]], dtype=torch.int64)
        audio_prompt = audio_prompt.unsqueeze(0).to("cuda:0")
        audio_prompt_len = audio_prompt_len.to("cuda:0")

        codes, codes_len = self.codec_encoder.encode_audio(audio_prompt, audio_prompt_len)

        actual_len = codes_len[0].item()
        return codes[0, :, :actual_len].to(torch.int32)

    def _process_context_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Process context codes: slice/repeat to target length and add special tokens.
        Same approach as C++ preprocessor - use beginning of audio for consistent voice.

        Note: If codes already contain BOS/EOS tokens (from new ONNX export),
        we strip them first, process the audio, then add back.
        """
        # Check if codes already have BOS/EOS tokens (from new ONNX export)
        # ctx_bos = 2018, ctx_eos = 2019
        has_bos_eos = codes[0, 0].item() == self.context_bos_token

        if has_bos_eos:
            # New ONNX format: codes = [ctx_bos, audio..., ctx_eos]
            # Strip BOS/EOS first, then process audio codes
            codes_cb0 = codes[0, :]
            eos_mask = codes_cb0 == self.context_eos_token
            if eos_mask.any():
                eos_idx = eos_mask.nonzero()[-1].item()
                # Audio is between BOS (idx 0) and EOS (idx eos_idx)
                codes = codes[:, 1:eos_idx]
            else:
                # Fallback: assume BOS at 0, rest is audio
                codes = codes[:, 1:]

        num_frames_to_slice = self.num_frames_to_slice

        if num_frames_to_slice < codes.shape[1]:
            # Use START of audio for consistent voice (same as C++ preprocessor)
            codes = codes[:, :num_frames_to_slice]
        else:
            # Repeat if shorter than target
            _num_repeats = int(np.ceil(num_frames_to_slice / codes.shape[1]))
            codes = codes.repeat(1, _num_repeats)[:, :num_frames_to_slice]

        # Add special tokens (context_bos, codes, context_eos, bos)
        context_bos = torch.full((8, 1), self.context_bos_token, dtype=codes.dtype, device=codes.device)
        context_eos = torch.full((8, 1), self.context_eos_token, dtype=codes.dtype, device=codes.device)
        bos_ids = torch.full((8, 1), self.bos_token, dtype=codes.dtype, device=codes.device)

        return torch.cat([context_bos, codes, context_eos, bos_ids], dim=1)

    def execute(self, requests):
        """`execute` is called for every inference request.
        Supports three modes:
        1. Multilingual mode: Uses pre-computed speaker embeddings
        2. Pre-encoded codes mode: Uses pre-encoded audio codes passed as input
        3. Zero-shot mode: Encodes raw audio prompt on-the-fly using CodecEncoder
        """
        for request in requests:
            input_ids = torch.from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "encoder_tokens").to_dlpack()
            ).squeeze(0)
            sentence_num = pb_utils.get_input_tensor_by_name(request, "sentence_num").as_numpy()[0]
            is_last_sentence = pb_utils.get_input_tensor_by_name(request, "is_last_sentence").as_numpy()[0]
            corr_id = request.correlation_id()
            response_sender = request.get_response_sender()
            request_num = pb_utils.get_input_tensor_by_name(request, "request_num").as_numpy()[0]
            is_last_request = pb_utils.get_input_tensor_by_name(request, "is_last_request").as_numpy()[0]
            context_audio_codes = None
            context_features = None

            # Check for zero-shot audio prompt
            audio_prompt_tensor = pb_utils.get_input_tensor_by_name(request, "audio_prompt")
            has_audio_prompt = audio_prompt_tensor is not None and audio_prompt_tensor.as_numpy().size > 1

            ## Zero-shot multilingual path - process audio prompt at runtime
            if self.is_zero_shot and has_audio_prompt and self.codec_encoder is not None:
                context_audio_codes, context_features = self._process_zero_shot_audio_prompt(
                    audio_prompt_tensor, request
                )
            ## Slice and repeat codes for multilingual path
            elif self.multilingual:
                speaker = pb_utils.get_input_tensor_by_name(request, "speaker").as_numpy()[0]
                if speaker is None:
                    speaker = -1
                context_features = self.context_map[speaker.item()]
                context_audio_codes = torch.zeros(
                    (8, context_features.shape[0]), dtype=torch.int32, device=context_features.device
                )
            else:
                codes = torch.from_dlpack(pb_utils.get_input_tensor_by_name(request, "codes").to_dlpack()).squeeze(0)
                num_frames_to_slice = self.num_frames_to_slice
                if num_frames_to_slice < codes.shape[1]:
                    start_idx = random.randint(0, codes.shape[1] - num_frames_to_slice)
                    codes = codes[:, start_idx : start_idx + num_frames_to_slice]
                else:
                    # Repeaet the audio if it is shorter than the desired duration
                    _num_repeats = int(np.ceil(num_frames_to_slice / codes.shape[1]))
                    # context_audio_codes is a tensor of shape (num_codebooks, T)
                    codes_repeated = codes.repeat(1, _num_repeats)
                    codes = codes_repeated[:, :num_frames_to_slice]

                context_bos_tensor = torch.full((8, 1), self.context_bos_token, dtype=codes.dtype, device=codes.device)
                context_eos_tensor = torch.full((8, 1), self.context_eos_token, dtype=codes.dtype, device=codes.device)
                bos_ids = torch.full((8, 1), self.bos_token, dtype=codes.dtype, device=codes.device)
                context_audio_codes = torch.cat([context_bos_tensor, codes, context_eos_tensor, bos_ids], dim=1)
                context_features = None

            # Create work item and enqueue (common for all paths)
            start_time = time.time()
            work_item = WorkItem(
                input_ids.cpu(),
                context_audio_codes,
                context_features,
                response_sender,
                sentence_num,
                is_last_sentence,
                corr_id,
                start_time,
                request_num,
                is_last_request,
            )
            self.encoder_queue.put(work_item)

    def _process_zero_shot_audio_prompt(
        self, audio_prompt_tensor, request
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process zero-shot audio prompt to generate context codes and features.

        Args:
            audio_prompt_tensor: Raw audio waveform tensor from the request
            request: The inference request object

        Returns:
            Tuple of (context_audio_codes, context_features)
        """
        # Get audio prompt as tensor
        audio_prompt = torch.from_dlpack(audio_prompt_tensor.to_dlpack())
        if audio_prompt.dim() == 1:
            audio_prompt = audio_prompt.unsqueeze(0)

        # Get audio length
        audio_length_tensor = pb_utils.get_input_tensor_by_name(request, "audio_prompt_length")
        if audio_length_tensor is not None:
            audio_length = torch.from_dlpack(audio_length_tensor.to_dlpack()).to(torch.int64)
        else:
            audio_length = torch.tensor([[audio_prompt.shape[-1]]], dtype=torch.int64)

        # Encode audio to codes using codec encoder
        codes, codes_len = self.codec_encoder.encode_audio(audio_prompt, audio_length)

        # Convert codes to INT64 (codec_encoder outputs INT32, decoder expects INT64)
        codes = codes.to(torch.int64)

        # codes shape: [batch, num_codebooks, frames]
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        # Truncate to max_context_frames
        num_frames = codes.shape[2]
        if num_frames > self.max_context_frames:
            # Truncate if longer than max
            codes = codes[:, :, : self.max_context_frames]
            num_frames = self.max_context_frames
        elif num_frames < self.max_context_frames:
            # Repeat audio codes to fill max_context_frames if shorter
            num_repeats = int(np.ceil(self.max_context_frames / num_frames))
            codes = codes.repeat(1, 1, num_repeats)[:, :, : self.max_context_frames]
            num_frames = self.max_context_frames

        # If we have a context encoder, generate context features
        if self.context_encoder is not None:
            # Context encoder now outputs embeddings with structure: [ctx_bos, audio, ctx_eos, audio_bos]
            context_features = self.context_encoder.encode_context(
                codes,
                self.max_context_frames,
                audio_bos_token=self.bos_token,  # Pass audio_bos_id for proper embedding
            )
            # Squeeze batch dimension if present
            if context_features.dim() == 3:
                context_features = context_features.squeeze(0)
            context_features = context_features.contiguous().to(torch.float16)

            # Build context_audio_codes with proper token structure: [ctx_bos, audio_codes, ctx_eos, audio_bos]
            # The decoder needs actual tokens even when using context_features
            codes = codes.squeeze(0)  # [num_codebooks, frames]
            context_bos_tensor = torch.full(
                (self.num_books, 1), self.context_bos_token, dtype=torch.int64, device=codes.device
            )
            context_eos_tensor = torch.full(
                (self.num_books, 1), self.context_eos_token, dtype=torch.int64, device=codes.device
            )
            bos_ids = torch.full((self.num_books, 1), self.bos_token, dtype=torch.int64, device=codes.device)
            context_audio_codes = torch.cat(
                [context_bos_tensor, codes.to(torch.int64), context_eos_tensor, bos_ids], dim=1
            )
        else:
            # No context encoder - use codes directly with BOS/EOS tokens
            codes = codes.squeeze(0)  # [num_codebooks, frames]

            context_bos_tensor = torch.full(
                (self.num_books, 1), self.context_bos_token, dtype=codes.dtype, device=codes.device
            )
            context_eos_tensor = torch.full(
                (self.num_books, 1), self.context_eos_token, dtype=codes.dtype, device=codes.device
            )
            bos_ids = torch.full((self.num_books, 1), self.bos_token, dtype=codes.dtype, device=codes.device)
            context_audio_codes = torch.cat([context_bos_tensor, codes, context_eos_tensor, bos_ids], dim=1)
            context_features = None

        return context_audio_codes, context_features

    def enqueue_work(self):
        enqueue_values_count = 10
        while self.running:
            # Use blocking get with timeout for better efficiency
            while not self.request_queue.empty():
                try:
                    work_item = self.request_queue.get(timeout=0.005)

                    convert_to_trtllm_request(
                        work_item,
                        self.num_books,
                        self.num_vocabs,
                        self.book_size,
                        self.temperature,
                        self.top_k,
                        self.pad_token,
                        self.eos_token,
                        self.cfg_scale,
                    )

                    with self.corr_id_map_lock:
                        if str(work_item.corr_id) not in self.corr_id_work_map:
                            self.corr_id_work_map[str(work_item.corr_id)] = WorkItemQueue(work_item.corr_id)
                        self.corr_id_work_map[str(work_item.corr_id)].add_workitem(work_item)
                except Exception as e:
                    self.logger.log_error(f"Error in enqueue_work: {e}")
                    raise e
                    break
                except Empty as e:
                    self.logger.log_error(f"Error Empty in enqueue_work: {e}")
                    break

            # with self.active_requests_lock:
            #     active_requests = self.active_requests
            active_requests = self.get_active_requests()
            if active_requests >= self.max_batch_size:
                time.sleep(0.01)  # Sleep longer when at capacity to reduce CPU usage
                continue
            # Get list of correlation IDs once per iteration
            with self.corr_id_map_lock:
                corr_id_to_delete = [
                    corr_id_queue.corr_id
                    for corr_id_queue in self.corr_id_work_map.values()
                    if corr_id_queue is not None and corr_id_queue.status == RequestStatus.FAILED
                ]
                for corr_id in corr_id_to_delete:
                    del self.corr_id_work_map[str(corr_id)]
                corr_id_list = [
                    corr_id_queue.corr_id
                    for corr_id_queue in self.corr_id_work_map.values()
                    if corr_id_queue is not None
                    and corr_id_queue.current_processing_item is None
                    and (corr_id_queue.status == RequestStatus.INIT or corr_id_queue.status == RequestStatus.WORK_DONE)
                ]
            if len(corr_id_list) == 0:
                time.sleep(0.002)  # 2ms polling
                continue
            # Process each correlation ID
            for corr_id in corr_id_list:
                # self.logger.log_info(f"Corr_id {corr_id_list} is being processed {time.perf_counter()}")
                # with self.active_requests_lock:
                #     active_requests = self.active_requests
                if not self.executor.can_enqueue_requests():
                    break
                active_requests = self.get_active_requests()
                if active_requests >= self.max_batch_size:
                    break
                work_item = None
                req_id = None
                try:
                    got_work_item, work_item = self.corr_id_work_map[str(corr_id)].get_next_workitem()
                    if not got_work_item:
                        continue
                    # self.logger.log_info(f"Corr_id {corr_id} is got from the work map {time.perf_counter()}")
                    # Pre-compute request parameters and enqueue to executor.
                    # Race fix per riva_magpie_fix_report.html §2.5: hold response_map_lock
                    # across enqueue + registration so a streaming response cannot land in
                    # await_generation before response_map[req_id] is set.
                    req_id = str(work_item.corr_id) + "_" + str(work_item.sentence_num.item())
                    work_item_queue = self.corr_id_work_map[str(corr_id)]
                    codes_to_append = None
                    read_idx = 0
                    write_idx = 0
                    if work_item.sentence_num.item() > 0:
                        codes_to_append = work_item_queue.audio_codes_history
                        read_idx = self.history_size
                        write_idx = self.history_size
                    with self.response_map_lock:
                        if not work_item.is_last_request.item():
                            req_id = self.executor.enqueue_request(work_item.trtllm_request)
                        with self.active_requests_lock:
                            self.active_requests.append(req_id)
                        work_item.status = RequestStatus.IN_PROGRESS
                        # Don't stringify codes_to_append tensor on hot path; use shape only.
                        self.logger.log_verbose(
                            f"Adding AudioDecoderWorkItem to response map: req_id={req_id} "
                            f"codes_to_append_shape={None if codes_to_append is None else tuple(codes_to_append.shape)}"
                        )
                        self.response_map[req_id] = AudioDecoderWorkItem(
                            corr_id,
                            self.chunk_size,
                            self.history_size,
                            self.future_size,
                            self.num_books,
                            self.book_size,
                            self.eos_token,
                            self.max_tokens,
                            req_id,
                            work_item.response_sender,
                            work_item.sentence_num,
                            work_item.is_last_sentence,
                            self.downsampling_factor,
                            self.logger,
                            self.sample_rate,
                            work_item.is_last_request,
                            stacking_factor=self.stacking_factor,
                            fade_ms=self.fade_ms,
                            read_idx=read_idx,
                            write_idx=write_idx,
                            audio_tokens=codes_to_append,
                        )
                    # self.logger.log_info(f"Corr_id {corr_id} is added to the response map {time.perf_counter()}")
                    if work_item.is_last_request.item():
                        if req_id in self.response_map:
                            self.response_map[req_id].process_trtllm_response(None)
                            with self.active_requests_lock:
                                if req_id in self.active_requests:
                                    self.active_requests.remove(req_id)
                except Exception as e:
                    raise e
                    if str(corr_id) in self.corr_id_work_map:
                        self.corr_id_work_map[str(corr_id)].mark_item_failed()
                        response = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                        work_item.response_sender.send(response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                        del self.corr_id_work_map[str(corr_id)]
                    if req_id is not None and req_id in self.active_requests:
                        with self.active_requests_lock:
                            self.active_requests.remove(req_id)

    def await_generation(self):
        while self.running:
            try:
                # Use optimized timeout for better responsiveness vs efficiency balance
                responses = self.executor.await_responses(
                    timeout=datetime.timedelta(milliseconds=50)
                )  # Further reduced timeout for better latency

                if not responses:
                    time.sleep(0.002)  # 2ms polling
                    continue

                for response in responses:
                    req_id = response.request_id
                    if response.result is not None and response.result.is_final and req_id in self.active_requests:
                        with self.active_requests_lock:
                            self.active_requests.remove(req_id)
                    with self.response_map_lock:
                        request = self.response_map.get(req_id, None)
                    if request is None:
                        self.executor.cancel_request(req_id)
                        if req_id in self.active_requests:
                            with self.active_requests_lock:
                                self.active_requests.remove(req_id)
                        continue
                    is_ok, corr_id = request.process_trtllm_response(response)
                    # self.logger.log_info(f"Corr_id {corr_id} TRTLLM response is processed {time.perf_counter()}")
                    # self.logger.log_info(f"Corr_id {corr_id} is_last_token_in: {request.is_last_token_in} {time.perf_counter()}")
                    # Cancel the executor as soon as EOS is observed at the token level so
                    # subsequent streaming chunks are not produced and appended to the buffer.
                    if request.is_last_token_in and req_id in self.active_requests:
                        self.executor.cancel_request(req_id)
                        with self.active_requests_lock:
                            if req_id in self.active_requests:
                                self.active_requests.remove(req_id)
                    if not is_ok:
                        # self.logger.log_info(f"Corr_id {corr_id} is deleted from the work map {time.perf_counter()}")
                        with self.corr_id_map_lock:
                            if str(corr_id) in self.corr_id_work_map:
                                del self.corr_id_work_map[str(corr_id)]
                        with self.response_map_lock:
                            if req_id in self.response_map:
                                del self.response_map[req_id]
                    if response.result is not None and response.result.is_final:
                        with self.corr_id_map_lock:
                            if str(corr_id) in self.corr_id_work_map:
                                self.corr_id_work_map[str(corr_id)].audio_codes_history = self.response_map[
                                    req_id
                                ].audio_codes[
                                    :,
                                    (self.response_map[req_id].write_idx - self.history_size - 2) : (
                                        self.response_map[req_id].write_idx - 2
                                    ),
                                ]
                    # Check if req_id still exists before accessing it
                    with self.response_map_lock:
                        if req_id in self.response_map and request.finished:
                            self.executor.cancel_request(req_id)
                            if req_id in self.active_requests:
                                with self.active_requests_lock:
                                    self.active_requests.remove(req_id)

            except Exception as e:
                self.logger.log_error(f"Error in await_generation main loop: {e}")
                raise e
                time.sleep(0.001)  # Small sleep to prevent tight loop on error

    def generate_audio(self):
        while self.running:
            try:
                # STEP 1: Read tokens from response_map and queue them to audio_decoder_queue
                # Get list of req_ids that have tokens, with proper locking to prevent race conditions
                req_list = []
                with self.response_map_lock:
                    req_list = list(self.response_map.values())
                work_items = []
                for request in req_list:
                    # Check if req_id still exists before accessing it
                    if request.has_tokens():
                        work_items.append(request)
                    elif request.send_final_audio:
                        request.send_response(None, 0)
                        with self.response_map_lock:
                            if str(request.corr_id) in self.corr_id_work_map:
                                self.corr_id_work_map[str(request.corr_id)].mark_item_completed()
                            if request.trtllm_request_id in self.response_map:
                                del self.response_map[request.trtllm_request_id]
                        with self.corr_id_map_lock:
                            if (
                                str(request.corr_id) in self.corr_id_work_map
                                and self.corr_id_work_map[str(request.corr_id)].status == RequestStatus.COMPLETED
                            ):
                                del self.corr_id_work_map[str(request.corr_id)]
                if len(work_items) == 0:
                    time.sleep(0.002)  # 2ms polling
                    continue

                audio_request_work_items = []
                audio_request = []
                history_lengths = []
                for work_item in work_items:
                    frames, history = None, 0
                    if work_item.has_tokens():
                        frames, history = work_item.read_tokens()
                    if frames is not None:
                        audio_request_work_items.append(frames)
                        history_lengths.append(history)
                        audio_request.append(work_item)
                    if len(audio_request_work_items) >= self.audio_decoder.max_batch_size:
                        break

                audio_response_work_items = self.audio_decoder.audio_decoder_infer(audio_request_work_items)

                for idx in range(len(audio_request)):
                    req_id = audio_request[idx].trtllm_request_id
                    try:
                        audio_request[idx].send_response(audio_response_work_items[idx], history_lengths[idx])
                        audio_request[idx].send_final_audio = audio_request[idx].is_last_token_in_read
                        # self.logger.log_info(f"Audio response for req_id {req_id} is sent {audio_response_work_items[idx].shape} Last token in read: {audio_request[idx].is_last_token_in_read} {time.perf_counter()}")

                    except Exception as e:
                        raise e
                        self.logger.log_error(
                            f"Error processing audio response for req_id {req_id}: {type(e).__name__}: {e}"
                        )

                        with self.response_map_lock:
                            if req_id in self.response_map:
                                del self.response_map[req_id]
                        with self.corr_id_map_lock:
                            if str(audio_request[idx].corr_id) in self.corr_id_work_map:
                                corr_id = audio_request[idx].corr_id
                                error = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                                self.corr_id_work_map[str(corr_id)].current_processing_item.response_sender.send(
                                    error, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                                )
                                self.corr_id_work_map[str(corr_id)].mark_item_failed()
                                del self.corr_id_work_map[str(corr_id)]
            except Exception as e:
                raise e
                self.logger.log_error(f"Error in audio response main loop: {e}")
                time.sleep(0.01)

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        with self.running_lock:
            self.running = False

        with self.encoder.running_lock:
            self.encoder.running = False

        # Stop codec encoder thread if running
        if hasattr(self, 'codec_encoder') and self.codec_encoder:
            with self.codec_encoder.running_lock:
                self.codec_encoder.running = False

        # Clean up encoder, audio decoder, and codec encoder
        if hasattr(self, 'encoder'):
            self.encoder.cleanup()
        if hasattr(self, 'audio_decoder'):
            self.audio_decoder.cleanup()
        if hasattr(self, 'codec_encoder') and self.codec_encoder:
            self.codec_encoder.cleanup()

        # Join threads with timeout to prevent hanging
        threads_to_join = [
            ("enqueue_work_thread", self.enqueue_work_thread),
            ("awaiter_thread", self.awaiter_thread),
            ("audio_generator_thread", self.audio_generator_thread),
            ("encoder_thread", self.encoder_thread),
        ]

        # Add codec encoder thread if it exists
        if hasattr(self, 'codec_encoder_thread') and self.codec_encoder_thread:
            threads_to_join.append(("codec_encoder_thread", self.codec_encoder_thread))

        for thread_name, thread in threads_to_join:
            if thread and thread.is_alive():
                self.logger.log_info(f"Joining {thread_name}...")
                thread.join(timeout=5.0)  # Add timeout to prevent indefinite hang
                if thread.is_alive():
                    self.logger.log_error(f"Warning: {thread_name} did not terminate within timeout")

        # Shutdown executor
        self.executor.shutdown()
        self.logger.log_info("Shutting down model")
