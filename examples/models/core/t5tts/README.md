
This describes how to run the t5tts in TRTLLM.
Build docker and compile TRTLLM as usual:

```bash
make -C docker build IMAGE_NAME=t5tts
make -C docker run LOCAL_USER=1 IMAGE_NAME=t5tts CONTAINER_NAME=t5tts
# 90-real - for H100
python3 ./scripts/build_wheel.py --cuda_architectures "90-real" --benchmarks --trt_root /usr/local/tensorrt
pip install build/tensorrt_llm-0.20.0rc0-cp312-cp312-linux_x86_64.whl
```


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



```