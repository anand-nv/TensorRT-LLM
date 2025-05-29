import json
import os
import time

import click
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import requests
import tensorrt as trt
import torch
from nemo.collections.tts.models import T5TTS_Model
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from omegaconf.omegaconf import OmegaConf, open_dict


def load_model(model_ckpt, model_cfg, audio_codec, engine_dir):
    with open_dict(model_cfg):
        model_cfg.codecmodel_path = audio_codec
        tokenizer_dir = os.path.join(engine_dir, 'tokenizer')
        os.makedirs(tokenizer_dir, exist_ok=True)
        if hasattr(model_cfg, 'text_tokenizer'):

            ipa_dict = "ipa_cmudict-0.7b_nv23.01.txt"
            heteronyms = "heteronyms-052722"
            files = [ipa_dict, heteronyms]
            url_root = "https://raw.githubusercontent.com/NVIDIA/NeMo/refs/heads/main/scripts/tts_dataset_files/"
            for fname in files:
                req = requests.get(f"{url_root}/{fname}")
                with open(os.path.join(tokenizer_dir, fname), 'wb') as file:
                    for chunk in req.iter_content(100000):
                        file.write(chunk)

            # Backward compatibility for models trained with absolute paths in text_tokenizer
            model_cfg.text_tokenizer.g2p.phoneme_dict = f"{tokenizer_dir}/{ipa_dict}"
            model_cfg.text_tokenizer.g2p.heteronyms = f"{tokenizer_dir}/{heteronyms}"
            model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0

        model_cfg.train_ds = None
        model_cfg.validation_ds = None
        model = T5TTS_Model(cfg=model_cfg)
        ckpt = torch.load(model_ckpt, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        print("Loaded weights.")

        model.use_kv_cache_for_inference = True

        model.cuda()
        model.eval()
        OmegaConf.save(model_cfg.text_tokenizers.english_phoneme,
                       os.path.join(tokenizer_dir, 'config.yaml'))

        return model


class MagpieEncoderExportTRT:

    def __init__(self,
                 checkpoint_dir,
                 engine_dir,
                 max_seq_len=256,
                 min_seq_len=3,
                 opt_seq_len=None,
                 minBS=1,
                 optBS=None,
                 maxBS=2,
                 dtype="float16"):
        self.checkpoint_dir = checkpoint_dir
        self.engine_dir = engine_dir

        self.encoder_config = {}

        self.dtype = dtype

        if opt_seq_len is None:
            opt_seq_len = min_seq_len + int((max_seq_len - min_seq_len) / 2)

        if optBS is None:
            optBS = minBS + int((maxBS - minBS) / 2)

        if opt_seq_len > max_seq_len or opt_seq_len < min_seq_len:
            raise Exception(
                f"Invalid opt_seq_len should be min_seq_len < opt_seq_len < max_seq_len "
            )

        if optBS > maxBS or optBS < minBS:
            raise Exception(f"Invalid optBS should be minBS < optBS < maxBS")

        self.in_feat_dim = None  # TODO: get from model
        self.num_tokens = None  # TODO: get from model

        self.min_seq_len = min_seq_len
        self.opt_seq_len = opt_seq_len

        self.max_seq_len = max_seq_len

        self.minBS = minBS
        self.optBS = optBS
        self.maxBS = maxBS

        self.encoder_config['min_seq_len'] = min_seq_len
        self.encoder_config['opt_seq_len'] = opt_seq_len
        self.encoder_config['max_seq_len'] = max_seq_len

        self.encoder_config['min_batch_size'] = minBS
        self.encoder_config['opt_batch_size'] = optBS
        self.encoder_config['max_batch_size'] = maxBS
        self.encoder_config['dtype'] = dtype

    def export_encoder_to_onnx(self, model, tokenizer_name="english_phoneme"):
        onnx_file = os.path.join(self.checkpoint_dir, 'encoder/encoder.onnx')
        texts = ["Hello world! How are you doing today?", "Second text"]
        text_encoding = [
            torch.IntTensor([model.bos_id] +
                            model.tokenizer.encode(text, tokenizer_name) +
                            [model.eos_id]).cuda() for text in texts
        ]
        text_lens = torch.IntTensor([len(text) for text in texts]).cuda()
        max_text_len = torch.max(text_lens).item()
        text_mask = get_mask_from_lengths(text_lens).cuda()  # (B, T)

        padded_text_encoding = stack_tensors(text_encoding,
                                             max_lens=[max_text_len],
                                             pad_value=model.tokenizer.pad)
        self.in_feat_dim = model.text_embedding.embedding_dim  #
        self.num_tokens = len(
            model.tokenizer.tokens) + 2  # add two for bos and eos
        self.encoder_config['num_tokens'] = self.num_tokens
        self.encoder_config['bos_id'] = self.num_tokens - 2
        self.encoder_config['eos_id'] = self.num_tokens - 1
        self.encoder_config['pad_id'] = model.tokenizer.pad

        texts = ["Hello world! How are you doing today?", "Second text"]
        text_encoding = [
            torch.IntTensor([model.bos_id] +
                            model.tokenizer.encode(text, tokenizer_name) +
                            [model.eos_id]).cuda() for text in texts
        ]
        text_lens = torch.IntTensor([len(text) for text in texts]).cuda()
        max_text_len = torch.max(text_lens).item()
        text_mask = get_mask_from_lengths(text_lens).cuda()  # (B, T)

        padded_text_encoding = stack_tensors(text_encoding,
                                             max_lens=[max_text_len],
                                             pad_value=model.tokenizer.pad)

        with torch.no_grad():

            embedded_text = model.text_embedding(padded_text_encoding)

            input_names = ["text", "text_mask"]
            output_names = ["output"]
            dynamic_axes = {
                "text": {
                    0: "batch_size",
                    1: "n_texts"
                },
                "text_mask": {
                    0: "batch_size",
                    1: "n_texts"
                },
            }
            print(f"{embedded_text.shape=}")
            inputs_args = {
                'x': embedded_text,
                'x_mask': text_mask,
                'cond': None,
                'cond_mask': None,
                'attn_prior': None,
                'multi_encoder_mapping': None
            }
            torch.onnx.export(model.t5_encoder,
                              inputs_args,
                              onnx_file,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes,
                              opset_version=17)

            # Add token embedding layer
            W = model.text_embedding.weight.data.cpu().numpy()
            W_emb = gs.Constant(name='w', values=W)
            no_categories = W.shape[0]
            token_dim = W.shape[1]
            depth = gs.Constant(name='depth', values=np.array(no_categories))
            values = gs.Constant(name='values',
                                 values=np.array([0.0, 1.0], dtype="float32"))

            x = gs.Variable(name='text_encoding',
                            dtype="int64",
                            shape=['batch_size', 'n_texts'])
            x_oh = gs.Variable(name='one_hot_text',
                               dtype="float32",
                               shape=['batch_size', 'n_texts', token_dim])
            text_embeddings = gs.Variable(
                name='text_embeddings',
                dtype="float32",
                shape=['batch_size', 'n_texts', token_dim])
            nodes = []
            nodes.append(
                gs.Node(op="OneHot",
                        name="onehot",
                        inputs=[x, depth, values],
                        outputs=[x_oh]))
            nodes.append(
                gs.Node(op="MatMul",
                        name="onehotemb",
                        inputs=[x_oh, W_emb],
                        outputs=[text_embeddings]))

            emb_graph = gs.Graph(nodes=nodes,
                                 inputs=[x],
                                 outputs=[text_embeddings],
                                 opset=17)

            emb_onnx = gs.export_onnx(emb_graph.cleanup())
            enc_onnx = onnx.load(onnx_file)
            emb_onnx.ir_version = enc_onnx.ir_version

            io_map = [('text_embeddings', 'text')]
            outputs = ["output"]
            merged_onnx = onnx.compose.merge_models(emb_onnx,
                                                    enc_onnx,
                                                    io_map,
                                                    outputs=outputs)
            onnx.save(merged_onnx, onnx_file)

    def generate_trt_engine(self):
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if self.dtype == "bfloat16":
            config.set_flag(trt.BuilderFlag.BF16)
        elif self.dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)

        #config.flags = config.flags
        parser = trt.OnnxParser(network, logger)
        onnx_file = os.path.join(self.checkpoint_dir, 'encoder/encoder.onnx')

        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
                print("Failed parsing %s" % onnx_file)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnx_file)

        nBS = -1
        nFeats = -1
        nMinBS = self.minBS
        nMaxBS = self.maxBS
        nOptBS = self.optBS

        input_feat = network.get_input(0)
        input_mask = network.get_input(1)
        input_feat.shape = [nBS, nFeats]
        input_mask.shape = [nBS, nFeats]
        profile.set_shape(
            input_feat.name,
            [nMinBS, self.min_seq_len],
            [nOptBS, self.opt_seq_len],
            [nMaxBS, self.max_seq_len],
        )
        profile.set_shape(
            input_mask.name,
            [nMinBS, self.min_seq_len],
            [nOptBS, self.opt_seq_len],
            [nMaxBS, self.max_seq_len],
        )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        plan_path = os.path.join(self.engine_dir, "encoder")
        os.makedirs(plan_path, exist_ok=True)

        plan_file = os.path.join(plan_path, 'encoder.plan')
        config_file = os.path.join(plan_path, 'config.json')

        if engineString == None:
            print("Failed building %s" % plan_file)
        else:
            print("Succeeded building %s in %d s" % (plan_file, t1 - t0))
            with open(plan_file, "wb") as f:
                f.write(engineString)
            with open(config_file, 'w') as jf:
                json.dump(self.encoder_config, jf)


@click.command()
@click.option("--model_ckpt", type=str, help="Path to model checkpoint")
@click.option("--audio_codec", type=str, help="Output Path to audio codec")
@click.option("--hparams_file", type=str, help="Path to hparams file")
@click.argument("tllm_checkpoint_dir", default="tllm_checkpoint", type=str)
@click.argument("engine_dir", default="engine", type=str)
def convert_encoder_to_trt(model_ckpt, audio_codec, hparams_file,
                           tllm_checkpoint_dir, engine_dir):
    model_cfg = OmegaConf.load(hparams_file).cfg
    model = load_model(model_ckpt, model_cfg, audio_codec, engine_dir)
    encoder = MagpieEncoderExportTRT(tllm_checkpoint_dir, engine_dir)
    encoder.export_encoder_to_onnx(model, tokenizer_name="english_phoneme")
    encoder.generate_trt_engine()


if __name__ == "__main__":
    convert_encoder_to_trt()
