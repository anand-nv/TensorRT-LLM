from tensorrt_llm.runtime import ModelRunnerCpp
import torch
import numpy as np
import os
import shutil


script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    np.set_printoptions(threshold=np.inf)  # Ensure full array is printed
    runner = ModelRunnerCpp.from_dir(
        engine_dir='newmodels/magpie_engine/decoder',
        is_enc_dec=False,
        max_input_len=512,
        rank=0,
        multi_block_mode=False,
        debug_mode=False,
        cross_kv_cache_fraction=0.5,
        kv_cache_free_gpu_memory_fraction=0.7,
    )

    encoder_encodings = torch.tensor(np.load("/code/tensorrt_llm/multilingual_inputs/text.npy")[0]).to(torch.float16)
    print(f"Encoder embeddings: {str(encoder_encodings.shape)}")

    books_num = 8
    book_size = 2024
    decoder_encodings = torch.tensor(np.load("/code/tensorrt_llm/multilingual_inputs/context.npy")[0]).to(torch.float16)
    print(f"Context embeddings: {str(decoder_encodings.shape)}")
    dummy_context_tokens = torch.tensor([0] * decoder_encodings.shape[0] * books_num, dtype=torch.int32)


    for run_idx in range(1):
        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=[dummy_context_tokens] * 16,
                encoder_input_features=[encoder_encodings] * 16,
                decoder_context_features=[decoder_encodings] * 16,
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