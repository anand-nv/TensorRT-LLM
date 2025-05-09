from tensorrt_llm.runtime import ModelRunnerCpp
import torch


def main():

    runner = ModelRunnerCpp.from_dir(
        engine_dir='newmodels/magpie_engine/',
        is_enc_dec=True,
        max_input_len=512,
        cross_kv_cache_fraction=0.5,
        rank=0,
    )

    encoder_tokens = torch.tensor([50] * 128, dtype=torch.int32)
    print(f"Encoder input tokens: {str(encoder_tokens)}")
    
    # Create batch_size=512 by repeating the same input
    batch_size = 1
    
    # Use [0] as decoder input (decoder start token) and repeat for batch_size
    decoder_input_ids = [0] * 16
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.int32)
    decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, 8)
    for i in range(8):
        decoder_input_ids[:, i] += 2048 * i
    decoder_input_ids = decoder_input_ids.flatten()
    print(f"Decoder input tokens: {str(len(decoder_input_ids))}")
    decoder_input_ids = [
        decoder_input_ids
        for _ in range(batch_size)
    ]
    
    # Use the tokenized input as encoder input and repeat for batch_size
    encoder_input_ids = [
        encoder_tokens
        for _ in range(batch_size)
    ]

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_ids=encoder_input_ids,
            max_new_tokens=32,
            end_id=1,
            pad_id=0,
            streaming=False,
        )
        torch.cuda.synchronize()

    output_ids = outputs.cpu().numpy()[0][0]
    print(f"Output tokens (len {len(output_ids)}):")
    print(output_ids)


if __name__ == "__main__":
    main()
