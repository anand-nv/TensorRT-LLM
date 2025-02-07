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
import time

import numpy as np
import torch
import json

from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)

from tensorrt_llm.bindings import GptJsonConfig, KVCacheType
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo

from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration,
                          MBartForConditionalGeneration,
                          T5ForConditionalGeneration)

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.runtime import EncDecModelRunner
from collections import OrderedDict


decoder_input_ids=[]
encoder_input_ids=[]

def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


def remove_tensor_padding(input_tensor, input_tensor_lengths=None, pad_value=0):
    if input_tensor.dim() == 2:
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] != pad_value
        ), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    elif input_tensor.dim() == 3:
        # Audio tensor case: batch, seq_len, feature_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"
        batch_size, seq_len, feature_len = input_tensor.shape

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(batch_size):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length, :])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)

    else:
        raise ValueError("Input tensor must have 2 or 3 dimensions")

    return output_tensor


class T5Decoding:
    def __init__(self, engine_dir, runtime_mapping, tokenizer, debug_mode=False):
        self.tokenizer = tokenizer
        self.decoder_config = read_config('decoder', engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode)
        self.dtype=str_dtype_to_torch(self.decoder_config['dtype'])
        self.pad_id=0
        self.audio_bos=2046
        self.audio_eos=2047
        self.context_bos=2044
        self.context_eos=2045
        self.text_bos=106337
        self.text_eos=106338

       
    @staticmethod
    def get_x_attention_mask(lens, max_length):
        batch_size = lens.shape[0]
        mask = torch.arange(max_length).repeat(batch_size, 1).to(lens.device) < lens[:, None]
        return mask

    def get_session(self, engine_dir, runtime_mapping, debug_mode=False):
        serialize_path = engine_dir / 'decoder' / 'rank0.engine'
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config['max_batch_size'],
            max_beam_width=self.decoder_config['max_beam_width'],
            num_heads=self.decoder_config['num_attention_heads'],
            num_kv_heads=self.decoder_config['num_attention_heads'],
            hidden_size=self.decoder_config['hidden_size'],
            vocab_size=self.decoder_config['vocab_size'],
            cross_attention=True,
            num_layers=self.decoder_config['num_hidden_layers'],
            gpt_attention_plugin=self.decoder_config['plugin_config']
            ['gpt_attention_plugin'],
            remove_input_padding=self.decoder_config['plugin_config']
            ['remove_input_padding'],
            kv_cache_type=KVCacheType.PAGED
            if self.decoder_config['plugin_config']['paged_kv_cache'] == True
            else KVCacheType.CONTINUOUS,
            has_position_embedding=self.
            decoder_config['has_position_embedding'],
            dtype=self.decoder_config['dtype'],
            has_token_type_embedding=False,
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode)

        return decoder_generation_session

    def generate(self,
                 decoder_input_ids,
                 encoder_outputs,
                 encoder_max_input_length,
                 encoder_input_lengths,
                 max_new_tokens=40,
                 num_beams=1):
        encoder_outputs = encoder_outputs.to(dtype=self.dtype)

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])], dtype=torch.int32, device='cuda'
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()

        cross_attention_mask = torch.ones([encoder_outputs.shape[0], 1, encoder_outputs.shape[1]]).int().cuda()

        # generation config
        sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_id, pad_id=self.tokenizer.pad_id, num_beams=num_beams
        )
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        if self.decoder_config['plugin_config']['remove_input_padding']:
            decoder_input_ids = remove_tensor_padding(
                decoder_input_ids, pad_value=self.tokenizer.pad_id)
            if encoder_outputs.dim() == 3:
                encoder_input_lengths = torch.full((encoder_outputs.shape[0],),
                                                   encoder_outputs.shape[1],
                                                   dtype=torch.int32,
                                                   device='cuda')

                encoder_outputs = remove_tensor_padding(encoder_outputs,
                                                        encoder_input_lengths)

        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids
