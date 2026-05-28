import os
import sys
import subprocess
import datetime
import json
from pathlib import Path

_script_dir = os.path.dirname(os.path.abspath(__file__))


def _ensure_c10_shim():
    _shim_so = os.path.join(_script_dir, "libc10_setdevice_shim.so")
    if not os.path.isfile(_shim_so):
        _src = os.path.join(_script_dir, "c10_setdevice_shim.cpp")
        if os.path.isfile(_src):
            try:
                import torch as _torch
                _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
                subprocess.run(
                    ["g++", "-shared", "-fPIC", "-o", _shim_so, _src,
                     "-L" + _torch_lib, "-lc10_cuda", "-Wl,-rpath," + _torch_lib],
                    check=True, timeout=60, cwd=_script_dir,
                )
            except Exception:
                return
    if os.path.isfile(_shim_so):
        import ctypes
        try:
            ctypes.CDLL(_shim_so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


import torch  # noqa: E402
_ensure_c10_shim()

import numpy as np
import soundfile as sf
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.bindings import executor as trtllm


def load_codec(codec_path):
    from nemo.collections.tts.models import AudioCodecModel
    cfg = AudioCodecModel.restore_from(codec_path, return_config=True)
    if hasattr(cfg, "use_scl_loss"):
        cfg.use_scl_loss = False
    codec = AudioCodecModel.restore_from(codec_path, strict=False, override_config_path=cfg)
    return codec.cuda().eval()


def codes_to_audio(codec, codes_np):
    C, T = codes_np.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32), int(codec.output_sample_rate)
    codes_t = torch.tensor(codes_np, dtype=torch.long, device="cuda").unsqueeze(0)
    codes_len = torch.tensor([T], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        audio, audio_len = codec.decode(tokens=codes_t, tokens_len=codes_len)
    return audio[0, : audio_len[0].item()].float().cpu().numpy(), int(codec.output_sample_rate)
# CategoricalSamplingPlugin is now compiled into nvinfer_plugin_tensorrt_llm.so
# and registered automatically when initTrtLlmPlugins() is called during executor init.


def decode_saved_codes_subprocess(codec_path: str, decode_jobs: list[tuple[str, str]]) -> None:
    """Decode saved codec-token .npy files in a fresh process, isolated from TRT-LLM shutdown."""
    if not decode_jobs:
        return

    script = r"""
import json
import os

import numpy as np
import soundfile as sf
import torch
from nemo.collections.tts.models import AudioCodecModel


def load_codec(codec_path):
    cfg = AudioCodecModel.restore_from(codec_path, return_config=True)
    if hasattr(cfg, "use_scl_loss"):
        cfg.use_scl_loss = False
    codec = AudioCodecModel.restore_from(codec_path, strict=False, override_config_path=cfg)
    return codec.cuda().eval()


def codes_to_audio(codec, codes_np):
    C, T = codes_np.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32), int(codec.output_sample_rate)
    codes_t = torch.tensor(codes_np, dtype=torch.long, device="cuda").unsqueeze(0)
    codes_len = torch.tensor([T], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        audio, audio_len = codec.decode(tokens=codes_t, tokens_len=codes_len)
    return audio[0, : audio_len[0].item()].float().cpu().numpy(), int(codec.output_sample_rate)


codec = load_codec(os.environ["T5TTS_CODEC_PATH"])
for npy_path, wav_path in json.loads(os.environ["T5TTS_DECODE_JOBS"]):
    codes = np.load(npy_path)
    audio, sr = codes_to_audio(codec, codes)
    sf.write(wav_path, audio, sr)
    print(f"  wrote {wav_path}  ({len(audio) / sr:.2f}s @ {sr} Hz)", flush=True)
"""
    env = os.environ.copy()
    env["T5TTS_CODEC_PATH"] = codec_path
    env["T5TTS_DECODE_JOBS"] = json.dumps(decode_jobs)
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


def _load_encoder_length(length_path: str | None, mask_path: str | None) -> int | None:
    """Load the true text length from a NeMo tensor dump, if available."""
    if length_path and os.path.exists(length_path):
        text_lens = torch.load(length_path, map_location="cpu")
        if isinstance(text_lens, torch.Tensor) and text_lens.numel() > 0:
            return int(text_lens.flatten()[0].item())

    if mask_path and os.path.exists(mask_path):
        text_mask = torch.load(mask_path, map_location="cpu")
        if isinstance(text_mask, torch.Tensor) and text_mask.numel() > 0:
            if text_mask.ndim >= 2:
                return int(text_mask[0].to(torch.int64).sum().item())
            return int(text_mask.to(torch.int64).sum().item())

    return None


def _trim_encoder_features(features: torch.Tensor, source_path: str) -> torch.Tensor:
    env_len = os.environ.get("T5TTS_ENCODER_LENGTH")
    if env_len:
        true_len = int(env_len)
        length_source = "T5TTS_ENCODER_LENGTH"
    else:
        source_dir = os.path.dirname(source_path) or "."
        env_len_path = os.environ.get("T5TTS_TEXT_LENS_PATH")
        env_mask_path = os.environ.get("T5TTS_TEXT_MASK_PATH")
        length_path = env_len_path or os.path.join(source_dir, "text_lens.pt")
        mask_path = env_mask_path or os.path.join(source_dir, "text_mask.pt")

        if not env_len_path and os.path.exists(length_path):
            source_mtime = os.path.getmtime(source_path)
            if source_mtime - os.path.getmtime(length_path) > 60:
                print(f"[warn] Ignoring stale text length sidecar {length_path}", flush=True)
                length_path = None
        if not env_mask_path and os.path.exists(mask_path):
            source_mtime = os.path.getmtime(source_path)
            if source_mtime - os.path.getmtime(mask_path) > 60:
                print(f"[warn] Ignoring stale text mask sidecar {mask_path}", flush=True)
                mask_path = None

        true_len = _load_encoder_length(length_path, mask_path)
        length_source = length_path if length_path and os.path.exists(length_path) else mask_path

    if true_len is None:
        print(f"[warn] No text_lens/text_mask found for {source_path}; using full encoder length {features.shape[0]}", flush=True)
        return features

    if true_len <= 0 or true_len > features.shape[0]:
        print(
            f"[warn] Ignoring invalid encoder length {true_len} from {length_source}; "
            f"using full encoder length {features.shape[0]}",
            flush=True,
        )
        return features

    if true_len < features.shape[0]:
        print(
            f"Trimmed encoder features from {features.shape[0]} to text length {true_len} "
            f"using {length_source}",
            flush=True,
        )
        return features[:true_len]

    print(f"Encoder text length {true_len} from {length_source}", flush=True)
    return features


def load_encoder_features() -> torch.Tensor:
    """Load the decoder cross-attention memory used by the exported TRT-LLM engine."""
    env_path = os.environ.get("T5TTS_ENCODER_FEATURES_PATH")
    candidates = [env_path] if env_path else []
    candidates.extend([
        "NeMo_main/text_encoder_out.pt",
        "text_encoder_out.pt",
        "NeMo_main/cond.pt",
        "cond.pt",
    ])
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.exists(candidate):
            features = torch.load(candidate)
            if features.ndim == 3:
                features = features[0]
            features = _trim_encoder_features(features, candidate)
            print(f"Encoder features: {candidate} {features.shape}", flush=True)
            return features.to(torch.float16)
    raise FileNotFoundError(
        "No encoder feature tensor found. Set T5TTS_ENCODER_FEATURES_PATH or provide text_encoder_out.pt/cond.pt."
    )


def load_decoder_context_features() -> torch.Tensor:
    """Load decoder prefix embeddings, including the appended audio BOS embedding."""
    env_path = os.environ.get("T5TTS_DECODER_CONTEXT_PATH")
    candidates = [env_path] if env_path else []
    candidates.extend([
        "NeMo_main/additional_decoder_input.pt",
        "additional_decoder_input.pt",
    ])
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.exists(candidate):
            features = torch.load(candidate)
            if features.ndim == 3:
                features = features[0]
            print(f"Decoder context features: {candidate} {features.shape}", flush=True)
            return features.to(torch.float16)
    raise FileNotFoundError(
        "No decoder context tensor found. Set T5TTS_DECODER_CONTEXT_PATH or provide additional_decoder_input.pt."
    )


def build_executor(
    engine_dir: str,
    *,
    max_batch_size: int | None = None,
    max_beam_width: int | None = None,
    max_num_tokens: int | None = None,
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
    if max_num_tokens is not None:
        executor_config.max_num_tokens = max_num_tokens
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
    seed: int | None = None,
    max_attend_count: int = 8,
    max_end_attend_count: int = 1_000_000,
    streaming: bool = False,
) -> torch.Tensor:
    """Enqueue one request per batch row; return padded int32 tensor (batch, 1, max_seq_len) on CUDA."""
    sampling = trtllm.SamplingConfig(
        beam_width=1,
        top_k=top_k,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )
    if seed is not None:
        sampling.seed = seed
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
                # NeMo infer_magpie.py defaults to ignore_finished_sentence_tracking=True:
                # attention-prior focus may reach the text end, but that alone does not
                # terminate generation. Keep the sink-escape behavior, and make end-focus
                # termination opt-in via T5TTS_MAX_END_ATTEND_COUNT.
                max_attend_count=max_attend_count,
                max_end_attend_count=max_end_attend_count,
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

    # Pad ragged sequences to the same length so torch.tensor gets a rectangular array.
    max_len = max(len(seq) for seq in output_ids)
    for seq in output_ids:
        pad_count = max_len - len(seq)
        if pad_count > 0:
            chunk_size = len(seq[0]) if seq else 8
            seq.extend([[end_id] * chunk_size] * pad_count)

    return torch.tensor(output_ids, dtype=torch.int32, device=cuda_device)


def main():
    np.set_printoptions(threshold=np.inf)

    engine_dir = os.environ.get(
        "T5TTS_DECODER_ENGINE_DIR",
        #"/home/siddhartht/nemo_models/magpie_may26/final_ckpt/models/magpie_tts-Magpie-Multilingual/1/",
        "/data/models_r/magpie_tts-Magpie-Multilingual/1/",
    )
    if not os.path.isdir(engine_dir):
        raise FileNotFoundError(
            f"Decoder engine dir not found: {engine_dir}. "
            "Build the decoder engine first or set T5TTS_DECODER_ENGINE_DIR."
        )

    json_config = GptJsonConfig.parse_file(Path(engine_dir) / "config.json")
    max_seq_len = json_config.model_config.max_seq_len

    multi_block_env = os.environ.get("T5TTS_MULTI_BLOCK_MODE", "1").lower()
    multi_block_mode = multi_block_env not in ("0", "false", "no", "off")
    max_num_tokens_env = os.environ.get("T5TTS_MAX_NUM_TOKENS")
    max_num_tokens = int(max_num_tokens_env) if max_num_tokens_env else None
    print(f"Executor multi_block_mode={multi_block_mode}", flush=True)
    executor = build_executor(engine_dir, multi_block_mode=multi_block_mode, max_num_tokens=max_num_tokens)
    if not executor.can_enqueue_requests():
        raise RuntimeError("This rank cannot enqueue requests (expected rank 0 for single-GPU).")

    num_vocabs = 16  # must match engine's num_vocabs so mPromptLen = ctx_positions*num_codebooks*stacking / 16 = 218

    # decoder_context_tts uses text_encoder_out as the decoder cross-attention memory.
    # cond.pt is kept as an explicit override/fallback for older tensor dumps.
    encoder_encodings = load_encoder_features()

    books_num = 8

    # Load context with BOS appended (218 positions = 217 context + 1 BOS).
    # The BOS token embedding was pre-computed by compare_executor_vs_nemo.py / embed_audio_tokens.
    # Provide stacked [cond_features, uncond_features] so runtimeBuffers copies features for
    # BOTH CFG sequences (conditional uses real context, unconditional uses zeros — both share BOS).
    cond_features = load_decoder_context_features()  # (218, 768)
    uncond_features = torch.zeros_like(cond_features)
    uncond_features[-1] = cond_features[-1]  # Share BOS embedding at last position
    # Shape (436, 768): runtimeBuffers copies cond[0:218] and uncond[218:436].
    # Required when cfg_scale > 0; without it the uncond half is zero-masked → CFG
    # combines garbage → repetition / late EOS (see runtimeBuffers.cpp:704-711).
    decoder_encodings = torch.cat([cond_features, uncond_features], dim=0)

    print(f"Context embeddings: {decoder_encodings.shape} {encoder_encodings.shape}")
    ctx_positions = cond_features.shape[0]  # 218
    stacking_factor = 2  # num codec frames per decoder step (must match engine's num_vocabs=16=books_num*stacking_factor)
    # Engine built with num_vocabs=16 expects ctx_positions*16 flat tokens so the embedding
    # view/mean groups them as ctx_positions positions of 16 tokens each (matching NeMo's embed_audio_tokens).
    dummy_context_tokens = torch.tensor([0] * ctx_positions * books_num * stacking_factor, dtype=torch.int64)
    print(f"Dummy context tokens: {dummy_context_tokens.shape} {dummy_context_tokens.dtype}")
    book_size = 2024

    codec_path = os.environ.get(
        "T5TTS_CODEC_PATH",
        "/home/siddhartht/tts/speechLM/NeMo_2503/models/causal_codec/21fps_causal_codecmodel.nemo",
    )
    decode_audio = os.path.isfile(codec_path)
    decode_jobs: list[tuple[str, str]] = []
    if not decode_audio:
        print(f"[warn] codec not found at {codec_path}; skipping wav output")

    bs = int(os.environ.get("T5TTS_BATCH_SIZE", "2"))
    num_runs = int(os.environ.get("T5TTS_NUM_RUNS", "10"))
    max_new_tokens = int(os.environ.get("T5TTS_MAX_NEW_TOKENS", "440"))
    seed_env = os.environ.get("T5TTS_SEED", "1234")
    seed = int(seed_env) if seed_env else None
    max_attend_count = int(os.environ.get("T5TTS_MAX_ATTEND_COUNT", "8"))
    ignore_finished_env = os.environ.get("T5TTS_IGNORE_FINISHED_SENTENCE_TRACKING", "1").lower()
    ignore_finished_sentence_tracking = ignore_finished_env not in ("0", "false", "no", "off")
    default_max_end_attend_count = "1000000" if ignore_finished_sentence_tracking else "16"
    max_end_attend_count = int(os.environ.get("T5TTS_MAX_END_ATTEND_COUNT", default_max_end_attend_count))
    invalid_summaries = []
    print(
        f"Attention prior max_attend_count={max_attend_count} "
        f"max_end_attend_count={max_end_attend_count}",
        flush=True,
    )
    # Executor is intentionally constructed once above; this loop only enqueues new request batches.
    for run_idx in range(num_runs):
        with torch.no_grad():
            outputs = run_generate(
                executor,
                batch_input_ids=[dummy_context_tokens.clone() for _ in range(bs)],
                encoder_input_features=[encoder_encodings.clone() for _ in range(bs)],
                decoder_context_features=[decoder_encodings.clone() for _ in range(bs)],
                max_new_tokens=max_new_tokens,
                end_id=2017,
                pad_id=2017,
                temperature=0.6,
                top_k=80,
                cfg_scale=2.5,
                max_seq_len=max_seq_len,
                num_vocabs=num_vocabs,
                seed=seed,
                max_attend_count=max_attend_count,
                max_end_attend_count=max_end_attend_count,
                streaming=True,
            )
        print(f"DONE RUN {run_idx}", flush=True)

        output_ids = outputs.cpu().numpy()
        print(f"Output tokens {output_ids.shape}", flush=True)

        prefix_len = dummy_context_tokens.shape[0]
        stacking_factor = 2  # num codec frames per decoder step
        for bi in range(output_ids.shape[0]):
            batch_output_ids = output_ids[bi]
            batch_output_ids = batch_output_ids.reshape(-1, books_num)
            #print(f"Final output tokens shape in batch {bi}: {batch_output_ids.shape}", flush=True)
            # Each decoder step emits stacking_factor rows (one per codec frame).
            # Frame f uses audio_embeddings[f*books_num .. (f+1)*books_num-1], so
            # codebook i of frame f has vocab offset (i + f*books_num) * book_size.
            for f in range(stacking_factor):
                for i in range(books_num):
                    batch_output_ids[f::stacking_factor, i] -= book_size * (i + f * books_num)

            eos_token = 2017
            print("batch_output_ids: ", batch_output_ids, output_ids.shape)
            for row_idx in range(batch_output_ids.shape[0]):
                if eos_token in batch_output_ids[row_idx, :]:
                    print("RUnning into break", eos_token, batch_output_ids[row_idx, :])
                    batch_output_ids = batch_output_ids[:row_idx, :]
                    break

            invalid_mask = (batch_output_ids < 0) | (batch_output_ids >= book_size)
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                invalid_positions = np.argwhere(invalid_mask)[:16].tolist()
                invalid_values = [
                    int(batch_output_ids[row, col])
                    for row, col in invalid_positions
                ]
                invalid_summaries.append(
                    (run_idx, bi, invalid_count, invalid_positions,
                     invalid_values))
                print(
                    f"VALIDATION run={run_idx} batch={bi} status=FAIL "
                    f"invalid_count={invalid_count} sample_positions={invalid_positions} "
                    f"sample_values={invalid_values}",
                    flush=True,
                )
            else:
                print(
                    f"VALIDATION run={run_idx} batch={bi} status=PASS "
                    f"frames={batch_output_ids.shape[0]}",
                    flush=True,
                )

            print(f"Final output tokens shape after removing EOS in batch {bi}: {batch_output_ids.shape}", flush=True)
            codes = batch_output_ids.T  # (books_num, T)
            npy_path = f"output_ids_{run_idx}_{bi}.npy"
            np.save(npy_path, codes)

            if decode_audio:
                decode_jobs.append((npy_path, f"output_ids_{run_idx}_{bi}.wav"))

    executor.shutdown()

    if invalid_summaries:
        raise RuntimeError(
            f"Invalid post-offset codec tokens detected: {invalid_summaries[:8]}"
        )

    if decode_audio:
        decode_saved_codes_subprocess(codec_path, decode_jobs)


if __name__ == "__main__":
    main()
