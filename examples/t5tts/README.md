
# Build TRTLLM

This describes how to run the t5tts in TRTLLM.
Build docker and compile TRTLLM as usual:

```bash
make -C docker build IMAGE_NAME=t5tts
make -C docker run LOCAL_USER=1 IMAGE_NAME=t5tts CONTAINER_NAME=t5tts
# 90-real - for H100
python3 ./scripts/build_wheel.py --cuda_architectures "90-real" --benchmarks --trt_root /usr/local/tensorrt
pip install build/tensorrt_llm-0.20.0rc0-cp312-cp312-linux_x86_64.whl
```

# Build Engine

Convert the checkpoint and build the engine:
```bash
# required to pip install omegaconf
# md5sum newmodels/t5tts.ckpt: fb177acdc447af56c8bbfa9d17c75f45
python examples/models/core/t5tts/convert_checkpoint.py \
    --model_path newmodels/t5tts.ckpt --output_dir newmodels/t5tts_convert

trtllm-build --checkpoint_dir newmodels/t5tts_convert/encoder/ \
--output_dir newmodels/t5tts_engine/encoder \
--paged_kv_cache enable --moe_plugin disable --max_beam_width 1 \
--max_batch_size 256 --max_input_len 128 --gemm_plugin float16 \
--bert_attention_plugin float16 --gpt_attention_plugin float16 \
--remove_input_padding enable --use_paged_context_fmha enable

trtllm-build --checkpoint_dir newmodels/t5tts_convert/decoder \
	--output_dir newmodels/t5tts_engine/decoder \
	--moe_plugin disable \
	--max_beam_width 1 \
	--max_batch_size 64 \
	--max_input_len 192 \
	--max_seq_len 512 \
	--max_encoder_input_len 512 \
	--gemm_plugin float16 \
	--bert_attention_plugin float16 \
	--gpt_attention_plugin float16 \
	--remove_input_padding enable \
 	--use_paged_context_fmha enable
```

# Toy inference

Finally run the model on the dummy input:
```bash
python examples/models/core/t5tts/run.py
```

# Benchmark

gpt manager benchmark is modified to run benchmark with context for decoder.

```bash
# prepare dummy inputs for inference
# 128 - number of phonemes in avergage sentence
# 160 - context length in frames, corresponds to 160 / 21.5 = 7.44 seconds
# 640 - total sequence length in frames, means 640 - 160 = 480 frames of audio generated,
# which corresponds to 480 / 21.5 = 22.33 seconds
# 768 - batch_size * 3, measure performance on 3 batches at max utilization
python examples/models/core/enc_dec/prepare_benchmark.py --output benchmark.json \
    --samples 768 \
    --max_input_id 98 \
	--num_vocabs 8 \
	--input_len 128 0 128 128 \
	--context_len 160 0 160 160 \
	--output_len 640 0 640 640

# run benchmark using generated dummy inputs
./cpp/build/benchmarks/gptManagerBenchmark \
    --dataset benchmark.json \
    --output_csv res.csv \
    --max_batch_size 256 \
    --concurrency 256 \
    --streaming \
    --num_vocabs 8 \
    --enable_chunked_context \
    --encoder_engine_dir newmodels/t5tts_engine/encoder \
    --decoder_engine_dir newmodels/t5tts_engine/decoder 2>&1 > /dev/null

# print results from res.csv
python3 -c "import csv; f=open('res.csv'); r=csv.reader(f); h=next(r); v=next(r); [print(f'{h[i]:<50}: {v[i]}') for i in range(len(h))]"
```
