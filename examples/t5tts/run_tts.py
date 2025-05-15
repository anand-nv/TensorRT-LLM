# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import time
from collections import OrderedDict

import torch

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime import ModelRunnerCpp
from tensorrt_llm.runtime.session import Session, TensorInfo


def read_config(component, engine_dir):
    config_path = f"{engine_dir}/{component}/config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class MagpieEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)
        config = read_config('encoder', engine_dir)
        self.dtype = config['dtype']
        self.encoder_config = config

    def get_session(self, engine_dir):
        serialize_path = f"{engine_dir}/encoder/rank0.engine"
        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())
        return session

    def get_encoder_feature(self, text_encodings):

        encoder_input_lengths = torch.IntTensor(
            [text_encoding.shape[-1] for text_encoding in text_encodings])
        encoder_input_ids = torch.stack(text_encodings).flatten()

        output_list = [
            TensorInfo('input_ids', str_dtype_to_trt('int32'),
                       encoder_input_ids.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       encoder_input_lengths.shape),
        ]

        inputs = OrderedDict()
        inputs['input_ids'] = encoder_input_ids
        inputs['input_lengths'] = encoder_input_lengths
        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        encoder_output = outputs['encoder_output']
        return encoder_output

        #inputs['position_ids'] = position_ids
        return inputs


def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None,
                          pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] !=
            pad_value), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor


def unpack_tensors(input_tensors, input_tensor_lengths):
    output_tensors = []
    for i in range(len(input_tensors)):
        output_tensors.append(input_tensors[i, :input_tensor_lengths[i]])
    return output_tensors


def get_enc_dec_runner(args, is_enc_dec=True):
    if not is_enc_dec:
        engine_dir = args.engine_dir + "/decoder"
    else:
        engine_dir = args.engine_dir

    decoder_layers_pos_out = [
        f'decoder_layers.{i}.pos_ff_output' for i in range(12)
    ]
    [f'decoder_layers.{i}.output' for i in range(12)]
    decoder_layer_qcross_attention_output = [
        f'decoder_layers.{i}.qcross_attention_output' for i in range(12)
    ]
    decoder_layer_output_norm = [
        f'decoder_layers.{i}.output_norm' for i in range(12)
    ]
    decoder_layer_output_norm_residual = [
        f'decoder_layers.{i}.output_norm_residual' for i in range(12)
    ]

    debug_tensor_names = [
        'decoder_layers.0.pos_ff_output',
        'decoder_layers.0.qcross_attention_output', 'decoder_layer_0_output',
        'decoder_layer_1_output', 'encoder_output', 'decoder_input_ids',
        'embedding_layer_output', 'logits_before_lmhead', 'logits'
    ]

    runner_kwargs = dict(
        engine_dir=engine_dir,
        debug_tensor_names=debug_tensor_names,
        debug_mode=args.debug_mode,
        is_enc_dec=is_enc_dec,
        max_input_len=1024,
        cross_kv_cache_fraction=0.5,
        rank=0,
    )
    return ModelRunnerCpp.from_dir(**runner_kwargs)


def evaluate_dec(args,
                 batch_input_ids,
                 return_dict=True,
                 eos_token_id=2047,
                 max_new_tokens=1024,
                 bos_token_id=2046,
                 pad_token_id=0):
    tllm_model = get_enc_dec_runner(args, is_enc_dec=False)
    config = read_config('decoder', args.engine_dir)

    #encoder=MagpieEncoding(args.engine_dir)
    batch_size = len(batch_input_ids)
    #encoder_outputs = encoder.get_encoder_feature(encoder_input_ids)

    encoder_io = torch.load('datafiles/encoder_io.pt')
    encoder_outputs = [
        encoder_io['output'][0, ::].to(device='cuda',
                                       dtype=str_dtype_to_torch(
                                           config['dtype']))
    ]
    encoder_output_lengths = [encoder_outputs[0].shape[0]]
    decoder_max_input_length = len(batch_input_ids[0]) + max_new_tokens
    #encoder_max_input_length = encoder_outputs[0].shape[0]

    #cross_attention_mask = torch.ones([
    #        batch_size, decoder_max_input_length + max_new_tokens,
    #        encoder_max_input_length
    #    ]).int().cuda()
    cross_attention_masks = [
        torch.ones([decoder_max_input_length, encoder_output_lengths[i]],
                   dtype=torch.int,
                   device='cuda') for i in range(batch_size)
    ]
    """decoder_input_ids = unpack_tensors(decoder_input_ids,
                                               decoder_input_lengths)
    encoder_outputs = unpack_tensors(encoder_outputs,
                                             encoder_output_lengths)
    """

    tllm_output = tllm_model.generate(
        batch_input_ids=batch_input_ids,
        encoder_input_features=encoder_outputs,
        encoder_input_lengths=encoder_output_lengths,
        encoder_output=encoder_outputs,
        #cross_attention_masks=cross_attention_masks,
        max_new_tokens=max_new_tokens,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        streaming=False,
        return_dict=return_dict,
    )
    return tllm_output


def evaluate_enc_dec(args,
                     batch_input_ids,
                     encoder_input_ids,
                     return_dict=True,
                     eos_token_id=2047,
                     max_new_tokens=1024,
                     bos_token_id=2046,
                     pad_token_id=0):
    tllm_model = get_enc_dec_runner(args, is_enc_dec=True)

    tllm_output = tllm_model.generate(
        batch_input_ids=batch_input_ids,
        encoder_input_ids=encoder_input_ids,
        max_new_tokens=max_new_tokens,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        streaming=False,
        return_dict=return_dict,
    )
    return tllm_output


def evaluate(args):
    # She had her dark suit in greasy wash water all year.
    audio_context_num_tokens = 2048
    audio_num_codebooks = 8
    #batch_size = 1
    eos_token_id = 2047

    text_encodings = [[
        96, 40, 29, 26, 93, 90, 55, 74, 52, 93, 29, 26, 39, 93, 90, 52, 77, 85,
        58, 93, 90, 64, 66, 65, 93, 84, 61, 93, 28, 39, 26, 22, 40, 46, 93, 90,
        68, 77, 86, 93, 44, 22, 41, 26, 39, 93, 90, 78, 59, 93, 90, 57, 84, 85,
        97
    ]]
    """, [
        96, 90, 86, 56, 93, 90, 55, 74, 52, 93, 90, 55, 82, 93, 25, 22, 39, 32,
        93, 90, 64, 66, 65, 93, 84, 61, 93, 90, 83, 85, 56, 64, 56, 93, 44, 22,
        40, 29, 93, 90, 68, 78, 65, 80, 93, 90, 78, 59, 93, 90, 57, 84, 85, 7,
        97
    ]]"""
    text_encodings = [
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
        25,
        22,
        39,
        32,
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
        44,
        22,
        40,
        29,
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
        7,
    ]

    batch_size = 1
    text_encodings = torch.IntTensor(text_encodings)

    audio_context = torch.load(
        'tllm_debug/context_codes_bos_scaled.pt').flatten().cuda()

    #encoder = MagpieEncoding(args.engine_dir)
    #encoder_output = encoder.get_encoder_feature()
    #torch.save(encoder_output, "encoder_output.pt")

    #inference_dtype = tllm_model.encoder_model_config.dtype
    batch_input_ids = [audio_context]
    print(f"{audio_context.shape=}")
    encoder_input_ids = [text_encodings]
    tik = time.time()
    return_dict = True
    if args.decoder_only:
        tllm_output = evaluate_dec(args,
                                   batch_input_ids,
                                   return_dict=True,
                                   eos_token_id=2047,
                                   max_new_tokens=1024,
                                   bos_token_id=2046,
                                   pad_token_id=0)
    else:
        tllm_output = evaluate_enc_dec(args,
                                       batch_input_ids,
                                       encoder_input_ids,
                                       return_dict=True,
                                       eos_token_id=2047,
                                       max_new_tokens=1024,
                                       bos_token_id=2046,
                                       pad_token_id=0)
    #tllm_output = evaluate_enc_dec(args, batch_input_ids, encoder_input_ids,
    #                              return_dict=True, eos_token_id = 2047, max_new_tokens=1024, bos_token_id=2046, pad_token_id=0)
    torch.save(tllm_output, "tllm_debug/tllm_output.pt")
    torch.cuda.synchronize()
    tok = time.time()
    batch_size = len(batch_input_ids)
    if return_dict:
        tllm_output_ids = tllm_output['output_ids']
    else:
        tllm_output_ids = tllm_output
    tllm_output_ids = tllm_output_ids % audio_context_num_tokens

    if tensorrt_llm.mpi_rank() == 0:
        __output_ids__ = tllm_output_ids.reshape(tllm_output_ids.shape[0], -1,
                                                 audio_num_codebooks)

        output_ids_is_eos = torch.where(__output_ids__ == eos_token_id, 1, 0)

        trim_output_idx = torch.argmin(torch.where(
            torch.sum(output_ids_is_eos, dim=-1) == 8, 0, 1),
                                       dim=1)
        for i in range(batch_size):
            if trim_output_idx[i] == 0:
                trim_output_idx[i] = __output_ids__.shape[1]

            trim_output_idx[i] = min(trim_output_idx[i] + 1,
                                     __output_ids__.shape[1])

        output_ids = [
            __output_ids__[i, :trim_output_idx[i], :] for i in range(batch_size)
        ]

        print("--------------------------------------")
        print(f"TRT-LLM E2E time {(tok-tik)*1000}ms")
        print("--------------------------------------")
    torch.save(output_ids, "tllm_debug/output_ids.pt")
    return output_ids


def print_tensor(tensor_name, tensor, num_elements=10):
    if tensor.dtype in (torch.int32, torch.int64):
        tensor = tensor.to(dtype=float)
    print(
        f'{tensor_name}: mean={tensor.abs().mean().item():.3f}, sum={tensor.abs().sum().item():.3f}, max={tensor.abs().max().item():.3f}'
    )
    # Pass num_elements=-1 will print the whole tensor
    if num_elements < 0:
        num_elements = torch.numel(tensor)
    print(f'{tensor.flatten()[:num_elements]}')
    print("Tensor Shape: ", tensor.size())
    print("")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--engine_dir", "-i", type=str, default="engines")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name",
                        type=str,
                        help="HuggingFace model name or FairSeq model path",
                        default="t5tts")
    parser.add_argument("--num_beams",
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument("--debug_mode",
                        help="Whether or not to turn on the debug mode",
                        action='store_true')
    parser.add_argument("--compare_hf_fp32",
                        help="Compare results with HuggingFace FP32",
                        action='store_true')
    parser.add_argument('--lora_dir', type=str, default=None, nargs="+")
    parser.add_argument('--lora_task_uids', type=str, default=None, nargs="+")
    parser.add_argument("--output_npy",
                        type=str,
                        default=None,
                        help="Store input/output tensors C++ runtime testing")
    parser.add_argument("--decoder_only",
                        help="Use decoder only model",
                        default=False,
                        action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    logger.set_level(args.log_level)
    evaluate(args)
