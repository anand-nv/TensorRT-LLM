## convert checkpoint
```bash
#  python3 convert_checkpoint.py --model_path=models_nemo_2/model.ckpt --dtype float32
python3 convert_checkpoint.py --model_path=<path_to_nemo_checkpoint> --dtype float32

```

## build engine
```bash
OUTPUT_DIR=engines

#build encoder
trtllm-build --checkpoint_dir tllm_checkpoint/encoder --output_dir ${OUTPUT_DIR}/encoder --max_batch_size 1 --gemm_plugin disable --bert_attention_plugin float32  

#build decoder
trtllm-build --checkpoint_dir tllm_checkpoint/decoder --output_dir ${OUTPUT_DIR}/decoder --max_batch_size 1 --gemm_plugin disable  --max_input_len 3000 --max_seq_len=3000  --gpt_attention_plugin float32 --remove_input_padding disable --moe_plugin disable  --enable_debug_output --gpt_attention_plugin disable

```

