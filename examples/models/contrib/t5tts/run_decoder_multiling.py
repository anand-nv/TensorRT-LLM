from tensorrt_llm.runtime import ModelRunnerCpp
import torch
import numpy as np
import os
import shutil
import ctypes
plugin_library_path = "/code/tensorrt_llm/build/libcategorical_sampling_plugin.so"

script_dir = os.path.dirname(os.path.abspath(__file__))



def main():
    np.set_printoptions(threshold=np.inf)  # Ensure full array is printed
    runner = ModelRunnerCpp.from_dir(
        engine_dir='/home/siddhartht/nemo_models/lt_framestacking/cu_13/lt_21_1xl12_h12_dim384/engine_fp16/decoder',
        is_enc_dec=False,
        max_input_len=512,
        rank=0,
        multi_block_mode=False,
        debug_mode=False,
        cross_kv_cache_fraction=0.5,
        kv_cache_free_gpu_memory_fraction=0.7,
    )

    batch = torch.load("/home/siddhartht/tts/speechLM/NeMo_2503/batch_prepared.pt")
    encoder_encodings = batch["text_encoder_out"].to(torch.float16).squeeze(0)[:200, :]
    print(f"Encoder embeddings: {str(encoder_encodings.shape)}")

    books_num = 8
    book_size = 2024
    decoder_encodings = torch.load("/home/siddhartht/scp_data/cifs/home/riva_speech/model_files/t5_tts/t5_trtllm_multilingual/feb_2026/hi_en_vn.pt")[0][0].to(torch.float16)
    #decoder_encodings = torch.cat([decoder_encodings.unsqueeze(0), decoder_encodings.unsqueeze(0)], dim=0)
    
    
    print('\n\n\n\n\n\n\n\n\n')
    print(f"Context embeddings: {str(decoder_encodings.dtype)} {encoder_encodings.shape}")
    dummy_context_tokens = torch.tensor([0] * decoder_encodings.shape[0] * books_num, dtype=torch.int64)
    print(f"Dummy context tokens: {str(dummy_context_tokens.shape)} {dummy_context_tokens.dtype}")
    bs = 1
    
    for run_idx in range(1):
        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=[dummy_context_tokens] * bs,
                encoder_input_features=[encoder_encodings] * bs,
                decoder_context_features=[decoder_encodings] * bs,
                max_new_tokens=440,
                end_id=2017,
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
            eos_token = 2017
            for row_idx in range(batch_output_ids.shape[0]):
                if eos_token in batch_output_ids[row_idx, :]:
                    batch_output_ids = batch_output_ids[:row_idx, :]
                    break
        
            print(f"Final output tokens shape after removing EOS in batch {bi}: {batch_output_ids.shape}", flush=True)
            np.save(f"output_ids_{run_idx}_{bi}.npy", batch_output_ids.T)


if __name__ == "__main__":
    main()