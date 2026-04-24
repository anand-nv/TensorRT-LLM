# Same workspace / PyTorch ABI bootstrap as run_decoder_multiling.py
import os
import sys
import subprocess

_script_dir = os.path.dirname(os.path.abspath(__file__))
#_workspace_root = os.path.abspath(os.path.join(_script_dir, os.pardir, os.pardir, os.pardir, os.pardir))
#if os.path.isdir(os.path.join(_workspace_root, "tensorrt_llm")):
#    sys.path.insert(0, _workspace_root)


def _ensure_c10_shim():
    _shim_so = os.path.join(_script_dir, "libc10_setdevice_shim.so")
    if not os.path.isfile(_shim_so):
        _src = os.path.join(_script_dir, "c10_setdevice_shim.cpp")
        if os.path.isfile(_src):
            try:
                import torch as _torch
                _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
                subprocess.run(
                    [
                        "g++",
                        "-shared",
                        "-fPIC",
                        "-o",
                        _shim_so,
                        _src,
                        "-L" + _torch_lib,
                        "-lc10_cuda",
                        "-Wl,-rpath," + _torch_lib,
                    ],
                    check=True,
                    timeout=60,
                    cwd=_script_dir,
                )
            except Exception:
                return
    print("Shim so: ", _shim_so)
    if os.path.isfile(_shim_so):
        import ctypes
        try:
            ctypes.CDLL(_shim_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


import torch  # noqa: E402
_ensure_c10_shim()


import datetime
from pathlib import Path

import numpy as np
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.bindings import executor as trtllm


def build_executor(
    engine_dir: str,
    *,
    max_batch_size: int | None = None,
    max_beam_width: int | None = None,
    kv_cache_free_gpu_memory_fraction: float | None = 0.7,
    cross_kv_cache_fraction: float | None = 0.5,
    multi_block_mode: bool = True,
    request_stats_max_iterations: int = 1000,
) -> trtllm.Executor:
    engine_path = Path(engine_dir)
    json_config = GptJsonConfig.parse_file(engine_path / "config.json")
    model_config = json_config.model_config

    if max_batch_size is None:
        max_batch_size = model_config.max_batch_size
    if max_beam_width is None:
        max_beam_width = model_config.max_beam_width

    kv_cache_config = trtllm.KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction,
        cross_kv_cache_fraction=cross_kv_cache_fraction,
        runtime_defaults=json_config.runtime_defaults,
    )

    extended = trtllm.ExtendedRuntimePerfKnobConfig()
    extended.multi_block_mode = multi_block_mode

    executor_config = trtllm.ExecutorConfig(
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        kv_cache_config=kv_cache_config,
        decoding_config=trtllm.DecodingConfig(),
        peft_cache_config=trtllm.PeftCacheConfig(),
        batching_type=trtllm.BatchingType.INFLIGHT,
        request_stats_max_iterations=request_stats_max_iterations,
    )
    executor_config.extended_runtime_perf_knob_config = extended
    executor_config.parallel_config = trtllm.ParallelConfig(
        trtllm.CommunicationType.MPI,
        trtllm.CommunicationMode.LEADER,
        device_ids=None,
        orchestrator_config=None,
    )

    return trtllm.Executor(engine_path, trtllm.ModelType.DECODER_ONLY, executor_config)


def run_generate(
    executor: trtllm.Executor,
    *,
    batch_input_ids: list[torch.Tensor],
    encoder_input_features: list[torch.Tensor],
    decoder_context_features: list[torch.Tensor],
    max_new_tokens: int,
    end_id: int,
    pad_id: int,
    temperature: float,
    top_k: int,
    cfg_scale: float,
    max_seq_len: int,
    num_vocabs: int,
    streaming: bool = False,
) -> torch.Tensor:
    """Enqueue one request per batch row; return padded int32 tensor (batch, 1, max_seq_len) on CUDA."""
    sampling = trtllm.SamplingConfig(
        beam_width=1,
        top_k=top_k,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )
    output_config = trtllm.OutputConfig()

    requests = []
    for i in range(len(batch_input_ids)):
        #print("batch_input_ids[i].shape: ", batch_input_ids[i].shape)
        #print("encoder_input_features[i].shape: ", encoder_input_features[i].shape)
        #print("decoder_context_features[i].shape: ", decoder_context_features[i].shape)
        #print("num_vocabs: ", num_vocabs)
        #print("max_new_tokens: ", max_new_tokens)
        #print("streaming: ", streaming)
        #print("sampling_config: ", sampling)
        #print("output_config: ", output_config)
        #print("end_id: ", end_id)
        #print("pad_id: ", pad_id)
        requests.append(
            trtllm.Request(
                batch_input_ids[i].tolist(),
                max_new_tokens,
                streaming=streaming,
                sampling_config=sampling,
                output_config=output_config,
                end_id=end_id,
                pad_id=pad_id,
                encoder_input_features=encoder_input_features[i].contiguous(),
                decoder_context_features=decoder_context_features[i].contiguous(),
                num_vocabs=num_vocabs,
            )
        )
        
    #print("Request: ", trtllm.Request(
    #            batch_input_ids[i].tolist(),
    #            max_new_tokens,
    #            streaming=streaming,
    #            sampling_config=sampling,
    #            output_config=output_config,
    #            end_id=end_id,
    #            pad_id=pad_id,
    #            encoder_input_features=encoder_input_features[i].contiguous(),
    #            decoder_context_features=decoder_context_features[i].contiguous(),
    #            num_vocabs=num_vocabs,
    #        ))

    request_ids = executor.enqueue_requests(requests)
    rid_to_idx = {rid: idx for idx, rid in enumerate(request_ids)}

    # Accumulate streamed chunks the same way ModelRunnerCpp._fill_output does (beam_width == 1).
    output_ids = [[] for _ in request_ids]
    finished = set()

    while finished != set(request_ids):
        responses = executor.await_responses()
        for response in responses:
            if response.has_error():
                raise RuntimeError(response.error_msg)
            result = response.result
            print(f"Got a new response: {result.output_token_ids[0]}")
            batch_idx = rid_to_idx[response.request_id]
            if len(result.output_token_ids[0]) > 8:
                idx = 0
                while (idx<len(result.output_token_ids[0])):
                    output_ids[batch_idx].append(result.output_token_ids[0][idx:idx+8])
                    idx += 8
            else:
                output_ids[batch_idx].append(result.output_token_ids[0])
            if result.is_final:
                finished.add(response.request_id)

        #for stats_per_iter in executor.get_latest_request_stats():
        #    print(
        #        f"request_stats iter={stats_per_iter.iter}: {stats_per_iter.to_json_str()}",
        #        flush=True,
        #    )

    #for stats_per_iter in executor.get_latest_request_stats():
    #    print(
        #    f"request_stats (final) iter={stats_per_iter.iter}: {stats_per_iter.to_json_str()}",
        #    flush=True,
        #)

    cuda_device = torch.device("cuda")
    #for beams in output_ids:
    #    for token_ids in beams:
    #        token_ids.extend([end_id] * (max_seq_len - len(token_ids)))
    for i in range(len(output_ids[0])):
        print(output_ids[0][i])
    return torch.tensor(output_ids, dtype=torch.int32, device=cuda_device)


def main():
    np.set_printoptions(threshold=np.inf)

    engine_dir = os.environ.get(
        "T5TTS_DECODER_ENGINE_DIR",
        "/data/models/magpie_tts-Magpie-Multilingual/1",
    )
    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(
            f"Decoder engine dir not found: {engine_dir}. "
            "Build the decoder engine first or set T5TTS_DECODER_ENGINE_DIR."
        )

    json_config = GptJsonConfig.parse_file(Path(engine_dir) / "config.json")
    max_seq_len = json_config.model_config.max_seq_len

    executor = build_executor(engine_dir)
    if not executor.can_enqueue_requests():
        raise RuntimeError("This rank cannot enqueue requests (expected rank 0 for single-GPU).")

    num_vocabs = 8 #executor.get_num_vocabs()

    batch = torch.load("/home/siddhartht/tts/speechLM/NeMo_2503/batch_prepared.pt")
    encoder_encodings = batch["text_encoder_out"].to(torch.float16).squeeze(0)[:200, :]
    print(encoder_encodings.shape)

    books_num = 8
    decoder_encodings = torch.load("/home/siddhartht/nemo_models/spectral_codec/combined_embed.pt")[0][0].to(
        torch.float16
    )

    print(f"Context embeddings: {decoder_encodings.shape} {encoder_encodings.shape}")
    dummy_context_tokens = torch.tensor([0] * decoder_encodings.shape[0] * books_num, dtype=torch.int64)
    print(f"Dummy context tokens: {dummy_context_tokens.shape} {dummy_context_tokens.dtype}")
    book_size = 2024

    bs = 1
    for run_idx in range(1):
        with torch.no_grad():
            outputs = run_generate(
                executor,
                batch_input_ids=[dummy_context_tokens] * bs,
                encoder_input_features=[encoder_encodings] * bs,
                decoder_context_features=[decoder_encodings] * bs,
                max_new_tokens=440,
                end_id=2017,
                pad_id=2017,
                temperature=0.6,
                top_k=80,
                cfg_scale=2.5,
                max_seq_len=max_seq_len,
                num_vocabs=num_vocabs,
                streaming=True,
            )
        print(f"DONE RUN {run_idx}", flush=True)

        output_ids = outputs.cpu().numpy()
        print(f"Output tokens {output_ids.shape}", flush=True)

        prefix_len = dummy_context_tokens.shape[0]
        for bi in range(output_ids.shape[0]):
            batch_output_ids = output_ids[bi]
            batch_output_ids = batch_output_ids.reshape(-1, books_num)
            #print(f"Final output tokens shape in batch {bi}: {batch_output_ids.shape}", flush=True)
            for i in range(books_num):
                batch_output_ids[:, i] -= book_size * i

            eos_token = 2017
            print("batch_output_ids: ", batch_output_ids, output_ids.shape)
            for row_idx in range(batch_output_ids.shape[0]):
                if eos_token in batch_output_ids[row_idx, :]:
                    print("RUnning into break", eos_token, batch_output_ids[row_idx, :])
                    batch_output_ids = batch_output_ids[:row_idx, :]
                    break

            print(f"Final output tokens shape after removing EOS in batch {bi}: {batch_output_ids.shape}", flush=True)
            np.save(f"output_ids_{run_idx}_{bi}.npy", batch_output_ids.T)

    executor.shutdown()


if __name__ == "__main__":
    main()
