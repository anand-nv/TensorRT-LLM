import argparse
import configparser
import json
import logging
import os
import types
from datetime import datetime
from pathlib import Path

import safetensors
import torch

from tensorrt_llm.functional import (LayerNormPositionType, LayerNormType,
                                     MLPType)

dir_path = os.path.dirname(os.path.realpath(__file__))
LOGGER = logging.getLogger(__name__)

layernorm_type_map = {i.name: i.value for i in LayerNormType}
layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}
mlp_type_map = {i.name: i.value for i in MLPType}

TORCH_DTYPES = {
    'float32': torch.float32,
    'float64': torch.float64,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
    )
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument('engine_dir')
    args = parser.parse_args()
    return args


def copy_args_to_component_config(component_config, args):
    for arg in vars(args):
        setattr(component_config, arg, getattr(args, arg))
    return component_config


def parse_model_config(args, ):
    config = configparser.ConfigParser()

    config["encoder"] = {}
    config["decoder"] = {}

    config["encoder"]["num_heads"] = "12"
    config["encoder"]['d_model'] = "768"  #hidden_size
    config["encoder"]['d_ffn'] = "3072"  #ffn_hidden_size
    config["encoder"]['vocab_size'] = "98"  # used to be 106339 in the branch
    config["encoder"]['n_positions'] = "2048"
    config["encoder"]['has_position_embedding'] = "true"
    #config["encoder"]['has_token_type_embedding'] =
    config["encoder"]['layernorm_position'] = "pre_layernorm"

    config["encoder"]['layernorm_type'] = "LayerNorm"
    config["encoder"]['num_layers'] = "6"
    # config["encoder"]['d_model'] /config["encoder"]["num_heads"]
    config["encoder"]['d_kv'] = f"{int(768/12)}" 

    config["decoder"]["num_heads"] = "12"
    config["decoder"]['d_model'] = "768"  #hidden_size
    config["decoder"]['d_ffn'] = "3072"  #ffn_hidden_size
    config["decoder"]['vocab_size'] = "16192"  # 8 * 2024
    config["decoder"]['n_positions'] = "2048"
    config["decoder"]['has_position_embedding'] = "true"
    config["decoder"]['layernorm_position'] = "pre_layernorm"

    config["decoder"]['layernorm_type'] = "LayerNorm"
    config["decoder"]['num_layers'] = "12"
    config["decoder"]["num_vocabs"] = "8"

    # manually set q_scaling to offset attention scaling's effect.
    # TODO: modify kernels to control whether to disable attention scaling
    def get_offset_q_scaling(config):
        scaling = 1 / config.head_size**.5
        return scaling

    config["structure"] = dict()
    config["structure"]["t5_with_bias"] = "false"
    #config["structure"]["use_gated_activation"] = str(hf_model.encoder.config.is_gated_act)
    config["structure"]["position_embedding_type"] = "learned_absolute"
    config["structure"]["model_type"] = "T5TTS"

    def parse_t5_config_by_component(config, component, args):
        component_config = types.SimpleNamespace()
        component_config = copy_args_to_component_config(component_config, args)
        component_config.n_head = config.getint(component, 'num_heads')
        component_config.hidden_size = config.getint(component, 'd_model')
        component_config.head_size = component_config.hidden_size // component_config.n_head

        component_config.ffn_hidden_size = config.getint(component, 'd_ffn')
        component_config.vocab_size = config.getint(component, 'vocab_size')
        component_config.n_positions = config.getint(component,
                                                     'n_positions',
                                                     fallback=2048)

        component_config.has_position_embedding = config.getboolean(
            component, 'has_position_embedding',
            fallback=False)  # TODO: hardcoded here

        component_config.has_token_type_embedding = config.getboolean(
            component, 'has_token_type_embedding', fallback=False)
        component_config.has_embedding_layernorm = config.getboolean(
            component, 'has_embedding_layernorm', fallback=False)
        component_config.has_embedding_scale = config.getboolean(
            component, 'has_embedding_scale', fallback=False)
        component_config.q_scaling = get_offset_q_scaling(component_config)
        component_config.has_attention_qkvo_bias = config.getboolean(
            component, 'has_attention_qkvo_bias',
            fallback=False)  # TODO: hardcoded here
        component_config.has_mlp_bias = config.getboolean(component,
                                                          'has_mlp_bias',
                                                          fallback=False)
        component_config.has_model_final_layernorm = config.getboolean(
            component, 'has_model_final_layernorm', fallback=True)
        component_config.layernorm_eps = config.getfloat(component,
                                                         'layer_norm_epsilon',
                                                         fallback=1e-5)
        component_config.layernorm_position = layernorm_position_map[config.get(
            component, 'layernorm_position',
            fallback='pre_layernorm')]  # TODO: hardcoded here
        component_config.layernorm_type = layernorm_type_map[config.get(
            component, 'layernorm_type', fallback='RmsNorm')]
        component_config.hidden_act = config.get(component,
                                                 'dense_act_fn',
                                                 fallback="gelu")
        component_config.gated_act = config.getboolean(component,
                                                       'is_gated_act',
                                                       fallback=True)
        #component_config.mlp_type = mlp_type_map['GatedMLP' if component_config.gated_act else 'MLP']
        component_config.num_buckets = config.getint(
            component, 'relative_attention_num_buckets', fallback=0)
        component_config.max_distance = config.getint(
            component, 'relative_attention_max_distance', fallback=0)
        component_config.position_embedding_type = config.get(
            'structure', 'position_embedding_type')
        component_config.logits_dtype = config.get(component,
                                                   'logits_dtype',
                                                   fallback='float16')

        if component == 'encoder':
            component_config.n_layer = config.getint(component, 'num_layers')

            component_config.relative_attention = config.get(
                'structure', 'position_embedding_type') == 'relative'

        elif component == 'decoder':
            component_config.n_layer = config.getint(component, 'num_layers')
            component_config.has_lm_head_bias = config.getboolean(
                component,  # TODO: T5 with bias
                'has_lm_head_bias',
                fallback=True)
            component_config.relative_attention = config.getboolean(
                component, 'relative_attention', fallback=False)
            component_config.rescale_before_lm_head = config.getboolean(
                component,
                'tie_word_embeddings',
                fallback=False,
            )  # default is True (for T5), but False for Flan-T5
            component_config.encoder_hidden_size = config.getint(
                'encoder', 'd_model')
            component_config.encoder_num_heads = config.getint(
                'encoder', 'num_heads')
            component_config.encoder_head_size = config.getint(
                'encoder', 'd_kv')
            #FIXME: check what is the correct generation process for the given checkpoint
            component_config.decoder_start_token_id = config.getint(
                'decoder', 'decoder_start_token_id', fallback=106339 - 2)
            component_config.eos_token_id = config.getint('decoder',
                                                          'eos_token_id',
                                                          fallback=2048 - 1)
            bos_token_id = config.get('decoder',
                                      'bos_token_id',
                                      fallback=2048 - 2)
            # T5 does not have bos_token_id
            component_config.bos_token_id = int(
                bos_token_id) if bos_token_id != "None" else None
            component_config.pad_token_id = config.getint('decoder',
                                                          'pad_token_id',
                                                          fallback=0)
            
            vocab_size = config.getint('decoder', 'vocab_size')
            num_vocabs = config.getint('decoder', 'num_vocabs')
            component_config.vocab_sizes = [vocab_size // num_vocabs] * num_vocabs

        else:
            assert False, 'Unsupported component!'

        return component_config

    encoder_config = parse_t5_config_by_component(config, "encoder", args)
    decoder_config = parse_t5_config_by_component(config, "decoder", args)

    return encoder_config, decoder_config


def convert_t5tts_encoder(
    config,
    model_dict,
    quant_algo: str = None,
    prefix: str = "encoder",
):
    weights = {}
    weights['embedding.vocab_embedding.weight'] = model_dict[
        'text_embedding.weight'].contiguous()
    weights['embedding.position_embedding.weight'] = model_dict[
        f'{prefix}.position_embeddings.weight'].contiguous()

    num_layers = config.n_layer
    for i in range(num_layers):
        weights[f'encoder_layers.{i}.attention_layernorm.weight'] = model_dict[
            f'{prefix}.layers.{i}.norm_self.weight'].contiguous()
        weights[f'encoder_layers.{i}.attention.qkv.weight'] = model_dict[
            f'{prefix}.layers.{i}.self_attention.qkv_net.weight'].contiguous(
            )
        weights[f'encoder_layers.{i}.attention.dense.weight'] = model_dict[
            f'{prefix}.layers.{i}.self_attention.o_net.weight'].contiguous()
        weights[f'encoder_layers.{i}.pos_ff_layernorm.weight'] = model_dict[
            f'{prefix}.layers.{i}.norm_pos_ff.weight'].contiguous()
        weights[f'encoder_layers.{i}.pos_ff.proj.weight'] = model_dict[
            f'{prefix}.layers.{i}.pos_ff.proj.conv.weight'].unsqueeze(
                3).contiguous()
        weights[f'encoder_layers.{i}.pos_ff.o_net.weight'] = model_dict[
            f'{prefix}.layers.{i}.pos_ff.o_net.conv.weight'].unsqueeze(
                3).contiguous()

    weights['final_layernorm.weight'] = model_dict[
        f'{prefix}.norm_out.weight'].contiguous()

    return weights


def convert_t5tts_decoder(
    config,
    model_dict,
    quant_algo: str = None,
    prefix: str = "decoder",
):
    weights = {}
    weights['lm_head.weight'] = model_dict['final_proj.weight'].clone().contiguous()

    weights['embedding.position_embedding.weight'] = model_dict[
        f'{prefix}.position_embeddings.weight'].contiguous()

    embs = [model_dict[f'audio_embeddings.{i}.weight'] for i in range(len(config.vocab_sizes))]
    embs.append(torch.zeros(1, 768, dtype=embs[0].dtype, device=embs[0].device))
    # embeddings have shape (2024 x 768) * 8, pad them adding extra entry in vocab which expands to zeros
    # we dont change the config, instead we change usage of the embedding dim in the model definition
    weights[f'embedding.vocab_embedding.weight'] = torch.cat(embs, dim=0).contiguous()
    
    num_layers = config.n_layer
    for i in range(num_layers):
        weights[
            f'decoder_layers.{i}.self_attention_layernorm.weight'] = model_dict[
                f'{prefix}.layers.{i}.norm_self.weight'].contiguous()
        weights[f'decoder_layers.{i}.self_attention.qkv.weight'] = model_dict[
            f'{prefix}.layers.{i}.self_attention.qkv_net.weight'].contiguous(
            )
        weights[f'decoder_layers.{i}.self_attention.dense.weight'] = model_dict[
            f'{prefix}.layers.{i}.self_attention.o_net.weight'].contiguous()
        weights[
            f'decoder_layers.{i}.cross_attention_layernorm.weight'] = model_dict[
                f'{prefix}.layers.{i}.norm_xattn_query.weight'].contiguous()

        qkv_weight = torch.cat([
            model_dict[f'{prefix}.layers.{i}.cross_attention.q_net.weight'],
            model_dict[f'{prefix}.layers.{i}.cross_attention.kv_net.weight']
        ], dim=0).contiguous()

        weights[f'decoder_layers.{i}.cross_attention.qkv.weight'] = qkv_weight
        weights[f'decoder_layers.{i}.cross_attention.dense.weight'] = model_dict[
            f'{prefix}.layers.{i}.cross_attention.o_net.weight'].contiguous()
        weights[f'decoder_layers.{i}.pos_ff_layernorm.weight'] = model_dict[
            f'{prefix}.layers.{i}.norm_pos_ff.weight'].contiguous()
        weights[
            f'decoder_layers.{i}.cross_attention_memory_layernorm.weight'] = model_dict[
                f'{prefix}.layers.{i}.norm_xattn_memory.weight'].contiguous()
        weights[f'decoder_layers.{i}.pos_ff.proj.weight'] = model_dict[
            f'{prefix}.layers.{i}.pos_ff.proj.conv.weight'].unsqueeze(
                3).contiguous()
        weights[f'decoder_layers.{i}.pos_ff.o_net.weight'] = model_dict[
            f'{prefix}.layers.{i}.pos_ff.o_net.conv.weight'].unsqueeze(
                3).contiguous()

    weights['final_layernorm.weight'] = model_dict[
        f'{prefix}.norm_out.weight'].contiguous()

    component_save_dir = os.path.join(args.output_dir, "decoder")
    os.makedirs(component_save_dir, exist_ok=True)
    return weights


def get_obj_dict(obj):
    return obj.__dict__


def convert_checkpoint(args, model):

    saved_dir = Path(args.output_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)

    encoder_saved_dir = saved_dir / "encoder"
    encoder_saved_dir.mkdir(parents=True, exist_ok=True)
    decoder_saved_dir = saved_dir / "decoder"
    decoder_saved_dir.mkdir(parents=True, exist_ok=True)

    world_size = args.tp_size * args.pp_size

    kv_cache_quant_algo = None
    quant_algo = None

    encoder_config, decoder_config = parse_model_config(args, )

    additional_settings = ["gated_act"]

    tllm_encoder_config = {
        'architecture': "T5TTSEncoderModel",
        'dtype': args.dtype,
        'logits_dtype': encoder_config.logits_dtype,
        'num_hidden_layers': encoder_config.n_layer,
        'num_attention_heads': encoder_config.n_head,
        'hidden_size': encoder_config.hidden_size,
        'norm_epsilon': encoder_config.layernorm_eps,
        'vocab_size': encoder_config.vocab_size,
        'position_embedding_type': encoder_config.position_embedding_type,
        'hidden_act': encoder_config.hidden_act,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'max_position_embeddings': encoder_config.n_positions,
        'num_key_value_heads': encoder_config.n_head,
        'head_size': encoder_config.head_size,
        'has_position_embedding': encoder_config.has_position_embedding,
        'layernorm_type': encoder_config.layernorm_type,
        'has_attention_qkvo_bias': encoder_config.has_attention_qkvo_bias,
        'has_mlp_bias': encoder_config.has_mlp_bias,
        'has_model_final_layernorm': encoder_config.has_model_final_layernorm,
        'has_embedding_layernorm': encoder_config.has_embedding_layernorm,
        'has_embedding_scale': encoder_config.has_embedding_scale,
        'intermediate_size': encoder_config.ffn_hidden_size,
        'q_scaling': encoder_config.q_scaling,
        'layernorm_position': encoder_config.layernorm_position,
        'relative_attention': encoder_config.relative_attention,
        'max_distance': encoder_config.max_distance,
        'num_buckets': encoder_config.num_buckets,
        'model_type': "t5tts"
    }

    for additional_setting in additional_settings:
        if hasattr(encoder_config, additional_setting):
            tllm_encoder_config.update({
                additional_setting:
                getattr(encoder_config, additional_setting)
            })

    tllm_decoder_config = {
        'architecture': "T5TTSDecoderModel",
        'dtype': args.dtype,
        'logits_dtype': decoder_config.logits_dtype,
        'num_hidden_layers': decoder_config.n_layer,
        'num_attention_heads': decoder_config.n_head,
        'hidden_size': decoder_config.hidden_size,
        'norm_epsilon': decoder_config.layernorm_eps,
        'vocab_size': decoder_config.vocab_size,
        'vocab_sizes': decoder_config.vocab_sizes,
        'use_attention_prior': True,
        'position_embedding_type': decoder_config.position_embedding_type,
        'hidden_act': decoder_config.hidden_act,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'max_position_embeddings': decoder_config.n_positions,
        'head_size': decoder_config.head_size,
        'has_position_embedding': decoder_config.has_position_embedding,
        'layernorm_type': decoder_config.layernorm_type,
        'has_attention_qkvo_bias': decoder_config.has_attention_qkvo_bias,
        'has_mlp_bias': decoder_config.has_mlp_bias,
        'has_model_final_layernorm': decoder_config.has_model_final_layernorm,
        'has_embedding_layernorm': decoder_config.has_embedding_layernorm,
        'has_embedding_scale': decoder_config.has_embedding_scale,
        'intermediate_size': decoder_config.ffn_hidden_size,
        'q_scaling': decoder_config.q_scaling,
        'layernorm_position': decoder_config.layernorm_position,
        'relative_attention': decoder_config.relative_attention,
        'max_distance': decoder_config.max_distance,
        'num_buckets': decoder_config.num_buckets,
        'model_type': "t5tts",
        'rescale_before_lm_head': decoder_config.rescale_before_lm_head,
        'encoder_hidden_size': decoder_config.encoder_hidden_size,
        'encoder_num_heads': decoder_config.encoder_num_heads,
        'encoder_head_size': decoder_config.encoder_head_size,
        'skip_cross_kv': args.skip_cross_kv,
        'use_implicit_relative_attention': args.use_implicit_relative_attention,
        'decoder_start_token_id': decoder_config.decoder_start_token_id,
        'eos_token_id': decoder_config.eos_token_id,
        'bos_token_id': decoder_config.bos_token_id,
        'pad_token_id': decoder_config.pad_token_id,
        'cross_attention': True,  #  this has to be provided explicitely
    }
    for additional_setting in additional_settings:
        if hasattr(decoder_config, additional_setting):
            tllm_decoder_config.update({
                additional_setting:
                getattr(decoder_config, additional_setting)
            })

    def convert_and_save(component: str = "encoder", ):
        # call get_encoder_config or get_decoder_config according to component
        if component == "encoder":
            config = tllm_encoder_config
        else:
            config = tllm_decoder_config

        component_save_dir = os.path.join(args.output_dir, component)
        if not os.path.exists(component_save_dir):
            os.makedirs(component_save_dir)

        with open(os.path.join(component_save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4, default=get_obj_dict)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            config['quantization'].update({
                'has_zero_point': True,
            })

        quant_algo = None
        """
        plugin_weight_only_quant_type = None
        if args.use_weight_only and args.weight_only_precision == 'int8':
            plugin_weight_only_quant_type = torch.int8
            quant_algo = QuantAlgo.W8A16
        elif args.use_weight_only and args.weight_only_precision == 'int4':
            plugin_weight_only_quant_type = torch.quint4x2
            quant_algo = QuantAlgo.W4A16
        elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            quant_algo = QuantAlgo.W4A16_GPTQ
        """

        if component == "encoder":

            weights = convert_t5tts_encoder(encoder_config,
                                            model_state_dict,
                                            quant_algo=quant_algo)
        else:
            assert component == "decoder"
            weights = convert_t5tts_decoder(decoder_config,
                                            model_state_dict,
                                            quant_algo=quant_algo)

        safetensors.torch.save_file(
            weights, os.path.join(component_save_dir, f'rank0.safetensors'))

    convert_and_save(component="encoder")
    convert_and_save(component="decoder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')

    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')

    parser.add_argument(
        "--workers",
        type=int,
        help="How many workers to spawn for conversion (default: 4)",
        default=4)

    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharding is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help=
        'Target inference dtype. Weights and Computation will be in this dtype, no matter what original dtype the weight checkpoint has.'
    )
    parser.add_argument(
        '--skip_cross_kv',
        action='store_true',
        help=
        'Skip redundant cross qkv computation by using TensorRT IfConditional switch (experimental).'
    )
    parser.add_argument(
        '--use_implicit_relative_attention',
        action='store_true',
        help=
        'Compute relative attention bias on the fly instead of pre-compute a relative attention bias table.'
    )
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()

    model_metadata = {}
    model_state_dict = torch.load(args.model_path,
                                  weights_only=False)['state_dict']
    for k in model_state_dict:
        model_state_dict[k] = model_state_dict[k].to(
            dtype=TORCH_DTYPES[args.dtype])
    convert_checkpoint(args, model_state_dict)

    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
