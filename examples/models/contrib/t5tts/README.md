
# Build TRTLLM

This describes how to run the t5tts in TRTLLM.
Build docker and compile TRTLLM as usual:

```bash
make -C docker build IMAGE_NAME=t5tts
make -C docker run LOCAL_USER=1 IMAGE_NAME=t5tts CONTAINER_NAME=t5tts
# 90-real - for H100
python3 ./scripts/build_wheel.py --cuda_architectures "90-real" --benchmarks --trt_root /usr/local/tensorrt
pip install build/tensorrt_llm-0.20.0rc0-cp312-cp312-linux_x86_64.whl

# add trtllm-build into a path
export PATH=/home/vklimkov/.local/bin/:$PATH

# for interactive python development, it helps to link working directory.
# if you dont plan doing any changes, no need for this
mv $HOME/.local/lib/python3.12/site-packages/tensorrt_llm $HOME/.local/lib/python3.12/site-packages/tensorrt_llm_bak
ln -s /code/tensorrt_llm/tensorrt_llm $HOME/.local/lib/python3.12/site-packages/tensorrt_llm
```

# Build Engine

Convert the checkpoint and build the engine:
```bash
# required to pip install omegaconf
# md5sum newmodels/t5tts.ckpt: fb177acdc447af56c8bbfa9d17c75f45
python examples/models/contrib/t5tts/convert_checkpoint.py \
    --model_path newmodels/t5tts.ckpt --output_dir newmodels/t5tts_convert

# optionally add "enable_debug_output"
trtllm-build --checkpoint_dir newmodels/t5tts_convert/decoder \
	--output_dir newmodels/t5tts_engine/decoder \
	--moe_plugin disable \
	--max_beam_width 1 \
	--max_batch_size 128 \
	--max_input_len 2048 \
	--max_seq_len 8192 \
	--max_encoder_input_len 256 \
	--gemm_plugin float16 \
	--bert_attention_plugin float16 \
	--gpt_attention_plugin float16 \
	--remove_input_padding enable
```

# Inputs to decoder

In order to run the MagpieTTS decoder, one needs text embeddings and decoder context.
You may set up entire end-to-end pipeline with codec and text encoder exported as TRT models,
but this is outside of TRTLLM scope. Here we assume that you run inference in NeMo and store
intermediate tensors to be passed to the decoder.

* Store context tokens or embeddings [here](https://github.com/NVIDIA/NeMo/blob/magpietts_2503_release/nemo/collections/tts/models/magpietts.py#L710)
* Store `context_tensors["cond"]` containing text embeddings [here](https://github.com/NVIDIA/NeMo/blob/magpietts_2503_release/nemo/collections/tts/models/magpietts.py#L1329)

# Toy inference

Finally run the model using intermediate tensors from NeMo:
```bash
python examples/models/core/t5tts/run_decoder.py
```
Adjust script to point to the inputs. Note `multi_block_mode=False` that is required for proper inference.
CFG is enabled by providing `cfg_scale` into `runner.generate`.
Produced audio tokens can be converted to audio using codec in NeMo.

# Benchmark

gpt manager benchmark is modified to run benchmark with context for decoder.

```bash
# prepare dummy inputs for inference
# 128 - number of phonemes in avergage sentence
# 160 - context length in frames, corresponds to 160 / 21.5 = 7.44 seconds
# 640 - total sequence length in frames, means 640 - 160 = 480 frames of audio generated,
# which corresponds to 480 / 21.5 = 22.33 seconds
python examples/models/contrib/t5tts/prepare_benchmark.py --samples 512 \
    --output benchmark.json --num_vocabs 8 --input_len 128 0 128 128 \
    --context_len 160 0 160 160 --output_len 480 0 480 480 \
    --text_emb_dim 768

# run benchmark using generated dummy inputs
# add "--cfg_scale=2.5" to run benchmark with cfg enabled, but you'd need to change concurrency to 32,
# to be batch_size / 2.
./cpp/build/benchmarks/gptManagerBenchmark --dataset benchmark.json  \
    --output_csv res.csv --max_batch_size 64  \
     --concurrency 64 --streaming --cross_kv_cache_fraction 0.5 \
    --engine_dir newmodels/magpie_engine/decoder --num_vocabs 8  \
    --multi_block_mode=false --kv_cache_free_gpu_mem_fraction=0.6 2>&1 > /dev/null

# print results from res.csv
python3 -c "import csv; f=open('res.csv'); r=csv.reader(f); h=next(r); v=next(r); [print(f'{h[i]:<50}: {v[i]}') for i in range(len(h))]"
```

# Configuration

Model hyperparameters are currently hardcoded in checkpoint conversion script.
Parameters of attention prior, which is enabled by default are [here](https://github.com/rmittal-github/TensorRT-LLM/blob/vklimkov/cfg_kernel_prior/examples/models/contrib/t5tts/convert_checkpoint.py#L414-L419).
Params of the decoder are [here](https://github.com/rmittal-github/TensorRT-LLM/blob/vklimkov/cfg_kernel_prior/examples/models/contrib/t5tts/convert_checkpoint.py#L100-L110).

