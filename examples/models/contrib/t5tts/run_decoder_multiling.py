# Prefer workspace tensorrt_llm over system install and fix PyTorch ABI mismatch
# (libth_common.so was built against PyTorch with c10::cuda::SetDevice(device, bool);
# current PyTorch has SetDevice(device). A shim provides the two-arg symbol.)
import os
import sys
import subprocess

_script_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_script_dir, os.pardir, os.pardir, os.pardir, os.pardir))
if os.path.isdir(os.path.join(_workspace_root, "tensorrt_llm")):
    sys.path.insert(0, _workspace_root)

# Load c10 SetDevice ABI shim after torch but before tensorrt_llm bindings,
# so libth_common finds _ZN3c104cuda9SetDeviceEab (two-arg) when it loads.
def _ensure_c10_shim():
    _shim_so = os.path.join(_script_dir, "libc10_setdevice_shim.so")
    if not os.path.isfile(_shim_so):
        _src = os.path.join(_script_dir, "c10_setdevice_shim.cpp")
        if os.path.isfile(_src):
            try:
                import torch as _torch
                _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
                subprocess.run(
                    ["g++", "-shared", "-fPIC", "-o", _shim_so, _src, "-L" + _torch_lib, "-lc10_cuda", "-Wl,-rpath," + _torch_lib],
                    check=True, timeout=60, cwd=_script_dir
                )
            except Exception:
                return
    if os.path.isfile(_shim_so):
        import ctypes
        try:
            ctypes.CDLL(_shim_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import torch  # noqa: E402 - must be before tensorrt_llm
_ensure_c10_shim()
from tensorrt_llm.runtime import ModelRunnerCpp
import numpy as np
import shutil
import ctypes
plugin_library_path = "/code/tensorrt_llm/build/libcategorical_sampling_plugin.so"

script_dir = os.path.dirname(os.path.abspath(__file__))



def main():
    np.set_printoptions(threshold=np.inf)  # Ensure full array is printed
    engine_dir = os.environ.get(
        "T5TTS_DECODER_ENGINE_DIR",
        "/data/models/magpie_tts-Magpie-Multilingual/1/",
    )
    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(
            f"Decoder engine dir not found: {engine_dir}. "
            "Build the decoder engine first (e.g. build_decoder.sh) or set T5TTS_DECODER_ENGINE_DIR to its path."
        )
    # max_input_len=None uses the engine's built-in limit. A fixed value smaller than the
    # encoder sequence length (encoder_input_features rows) can under-provision cross KV and
    # lead to allocator corruption in C++.
    print("Creating runner from engine directory: %s", engine_dir)
    runner = ModelRunnerCpp.from_dir(
        engine_dir=engine_dir,
        is_enc_dec=False,
        max_input_len=None,
        rank=0,
        multi_block_mode=True,
        debug_mode=False,
        cross_kv_cache_fraction=0.5,
        kv_cache_free_gpu_memory_fraction=0.7,
    )

    batch = torch.load("/home/siddhartht/tts/speechLM/NeMo_2503/batch_prepared.pt")
    encoder_encodings = batch["text_encoder_out"].to(torch.float16).squeeze(0)[:200, :]
    print(encoder_encodings.shape)
    #encoder_encodings = torch.cat([encoder_encodings, encoder_encodings, encoder_encodings, encoder_encodings], dim=0)
    print(f"Encoder embeddings: {str(encoder_encodings.shape)}")

    books_num = 8
    book_size = 1032
    decoder_encodings = torch.load("/home/siddhartht/nemo_models/spectral_codec/combined_embed.pt")[0][0].to(torch.float16)
    #decoder_encodings = torch.cat([decoder_encodings.unsqueeze(0), decoder_encodings.unsqueeze(0)], dim=0)
    
    
    print('\n\n\n\n\n\n\n\n\n')
    print(f"Context embeddings: {str(decoder_encodings.shape)} {encoder_encodings.shape}")
    dummy_context_tokens = torch.tensor([0] * decoder_encodings.shape[0] * books_num, dtype=torch.int64)
    print(f"Dummy context tokens: {str(dummy_context_tokens.shape)} {dummy_context_tokens.dtype}")
    bs = 2
    for run_idx in range(1):
        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=[dummy_context_tokens] * bs,
                encoder_input_features=[encoder_encodings] * bs,
                decoder_context_features=[decoder_encodings] * bs,
                max_new_tokens=440,
                end_id=1025,
                pad_id=2017,
                temperature=0.6,
                top_k=80,
                streaming=False,
                cfg_scale=2.5,
            )
        print(f"DONE RUN {run_idx}", flush=True)
        #shutil.move("/tmp/tllm_debug/PP_1/TP_1/", f"/tmp/tllm_debug/PP_1/TP_run_{run_idx}/")

        output_ids = outputs.cpu().numpy()
        print(f"Output tokens {output_ids.shape}", flush=True)

        # select first 3 frames
        # skip prefix
        for bi in range(output_ids.shape[0]):
            batch_output_ids = output_ids[bi][0][dummy_context_tokens.shape[0]:]
            batch_output_ids = batch_output_ids.reshape(-1, books_num)
            for i in range(books_num):
                batch_output_ids[:, i] -= book_size * i
            print(f"Final output tokens shape in batch {bi}: {batch_output_ids.shape}", flush=True)
            
            # find the occurrence of eos token and discard the rest
            eos_token = 1025
            for row_idx in range(batch_output_ids.shape[0]):
                if eos_token in batch_output_ids[row_idx, :]:
                    batch_output_ids = batch_output_ids[:row_idx, :]
                    break
        
            print(f"Final output tokens shape after removing EOS in batch {bi}: {batch_output_ids.shape}", flush=True)
            np.save(f"output_ids_{run_idx}_{bi}.npy", batch_output_ids.T)


if __name__ == "__main__":
    main()