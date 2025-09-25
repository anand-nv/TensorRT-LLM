from tensorrt_llm.runtime import ModelRunnerCpp
import torch
import numpy as np
import os
import shutil
import glob


script_dir = os.path.dirname(os.path.abspath(__file__))


out_dir = "/code/tensorrt_llm/predicted_tokens_grpo_checkpoint_withcfg"
os.makedirs(out_dir, exist_ok=True)

def main():
    np.set_printoptions(threshold=np.inf)  # Ensure full array is printed
    runner = ModelRunnerCpp.from_dir(
        engine_dir='newmodels/magpie_engine/decoder',
        is_enc_dec=False,
        max_input_len=512,
        rank=0,
        multi_block_mode=False,
        #debug_mode=True,
        cross_kv_cache_fraction=0.5,
        kv_cache_free_gpu_memory_fraction=0.7,
    )

    books_num = 8
    book_size = 2024
    decoder_input_ids = torch.tensor(np.load("/code/tensorrt_llm/encoder_outputs/decoder_context.npy"), dtype=torch.int32).transpose(1, 0)
    print(decoder_input_ids.shape)
    bos_ids = torch.tensor([2016] * books_num, dtype=torch.int32).unsqueeze(0)  # 1 x 8
    decoder_input_ids = torch.cat([decoder_input_ids, bos_ids], dim=0)  # 110 x 8

    for i in range(books_num):
        decoder_input_ids[:, i] += book_size * i
    print(f"Decoder input tokens: {str(decoder_input_ids.shape)}")
    decoder_input_ids = decoder_input_ids.flatten()

    for path in glob.glob("/code/tensorrt_llm/encoder_outputs/encoder_outputs_*.npy"):
        print(f"================",flush=True)
        encoder_encodings = torch.tensor(np.load(path)).to(torch.float16)
        print(f"Encoder input tokens: {str(encoder_encodings.shape)}", flush=True)

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids=[decoder_input_ids],
                encoder_input_features=[encoder_encodings],
                max_new_tokens=440,
                end_id=2017,
                pad_id=2017,
                temperature=0.6,
                top_k=80,
                streaming=False,
                cfg_scale=2.5,
            )

            output_ids = outputs.cpu().numpy()
            print(f"Output tokens {output_ids.shape}", flush=True)

            # select first 3 frames
            # skip prefix
            for bi in range(output_ids.shape[0]):
                batch_output_ids = output_ids[bi][0][decoder_input_ids.shape[0]:]
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
            name = os.path.basename(path).replace("encoder_outputs_", "predicted_tokens_")
            out_path = os.path.join(out_dir, name)
            np.save(out_path, batch_output_ids.T)
        print(f"========END=======",flush=True)


if __name__ == "__main__":
    main()
