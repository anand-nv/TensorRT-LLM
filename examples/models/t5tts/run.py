from tensorrt_llm.runtime import ModelRunnerCpp
import torch
import numpy as np
import os


script_dir = os.path.dirname(os.path.abspath(__file__))


def main():

    runner = ModelRunnerCpp.from_dir(
        engine_dir='newmodels/magpie_engine/',
        is_enc_dec=True,
        max_input_len=512,
        cross_kv_cache_fraction=0.5,
        rank=0,
    )

    encoder_tokens = torch.tensor(np.load(os.path.join(script_dir, "debug_io/text_tokens.npy"))[0], dtype=torch.int32)
    print(f"Encoder input tokens: {str(encoder_tokens)}")

    books_num = 8
    book_size = 2048
    decoder_input_ids = torch.tensor(np.load(os.path.join(script_dir, "debug_io/context_audio_codes.npy"))[0], dtype=torch.int32).transpose(1, 0)  # 8 x 109 -> 109 x 8
    bos_ids = torch.tensor([book_size - 2] * books_num, dtype=torch.int32).unsqueeze(0)  # 1 x 8
    decoder_input_ids = torch.cat([decoder_input_ids, bos_ids], dim=0)  # 110 x 8
    for i in range(books_num):
        decoder_input_ids[:, i] += book_size * i
    print(f"Decoder input tokens: {str(decoder_input_ids.shape)}")
    decoder_input_ids = decoder_input_ids.flatten()    

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=[decoder_input_ids],
            encoder_input_ids=[encoder_tokens],
            max_new_tokens=64,
            end_id=book_size - 1,
            pad_id=book_size - 1,
            streaming=False,
        )
        torch.cuda.synchronize()

    output_ids = outputs.cpu().numpy()
    print(f"Output tokens {output_ids.shape}", flush=True)

    # select first 3 frames
    # skip prefix
    output_ids = output_ids[0][0][(decoder_input_ids.shape[0] - books_num):]
    output_ids = output_ids.reshape(-1, books_num)
    for i in range(books_num):
        output_ids[:, i] -= book_size * i
    to_print = output_ids[:3, :]
    np.set_printoptions(threshold=np.inf)  # Ensure full array is printed
    print("First 3 frames:")
    print(to_print)


if __name__ == "__main__":
    main()
