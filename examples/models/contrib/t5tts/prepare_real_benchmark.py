import argparse
import json
import glob
from pathlib import Path
import numpy as np
import os

"""prepare_real_benchmark.py

Generate a workload JSON file that contains *real* requests for the T5TTS
pipeline.  Instead of synthesizing random tokens (as done by
prepare_benchmark.py) this script uses the same decoder‐side context tokens and
pre–computed encoder output features that are fed into `run_decoder.py`.

Each request in the produced JSON includes a unique "id" so that subsequent
benchmarking tools (e.g. gptManagerBenchmark with --dumpRequests) can map the
results back to the originating sample.

The JSON schema is compatible with the existing C++ helper
`parseWorkloadJson()`.  Unknown fields ("id", "encoder_features_path") are
ignored by the current parser but are handy for downstream tooling.
"""

# Constants taken from run_decoder.py
BOOKS_NUM_DEFAULT = 8
BOOK_SIZE_DEFAULT = 2024  # Size of each parallel vocabulary
BOS_TOKEN_ID_DEFAULT = 2016
OUTPUT_LEN_DEFAULT = 440  # Same value used in run_decoder_all.py

def build_context_ids(decoder_context: np.ndarray,
                      books_num: int = BOOKS_NUM_DEFAULT,
                      book_size: int = BOOK_SIZE_DEFAULT,
                      bos_token_id: int = BOS_TOKEN_ID_DEFAULT) -> list[int]:
    """Replicate the token manipulation logic from run_decoder.py.

    1. The incoming *decoder_context* is expected to be a 2-D array with shape
       (time, books_num) **after** transposition.
    2. A BOS token (``bos_token_id``) is appended as an extra time-step.
    3. The tokens in column *i* are offset by ``book_size * i`` so that all
       books occupy disjoint regions in the token space.
    4. The resulting matrix is flattened row-major.
    """

    # Append BOS row (1 x books_num)
    bos_row = np.full((1, books_num), bos_token_id, dtype=decoder_context.dtype)
    ctx = np.concatenate([decoder_context, bos_row], axis=0)  # shape (T+1, books_num)

    # Offset each column
    for i in range(books_num):
        ctx[:, i] += book_size * i

    # Flatten row-major and convert to native int (to avoid NumPy types in json)
    return [int(t) for t in ctx.flatten()]


def generate_samples(encoder_glob: str,
                     decoder_context_path: Path,
                     output_file: Path,
                     books_num: int = BOOKS_NUM_DEFAULT,
                     book_size: int = BOOK_SIZE_DEFAULT,
                     bos_token_id: int = BOS_TOKEN_ID_DEFAULT,
                     output_len: int = OUTPUT_LEN_DEFAULT) -> None:
    """Create the workload file.

    Parameters
    ----------
    encoder_glob:  Glob pattern that expands to the *.npy* files containing the
        *encoder_output* features for each utterance.
    decoder_context_path:  Path to the *.npy* file with decoder-side context
        tokens ( *before* the BOS row is appended and before shifting).
    output_file:  Where the JSON will be written.
    """

    encoder_paths = sorted(glob.glob(encoder_glob))
    if not encoder_paths:
        raise FileNotFoundError(f"No encoder output files matched '{encoder_glob}'.")

    # Decoder context is shared across all requests
    raw_context = np.load(decoder_context_path)
    # run_decoder loads the array and immediately transposes: (B, T) -> (T, B)
    raw_context = raw_context.T  # ensure shape aligns with run_decoder logic
    context_ids = build_context_ids(raw_context, books_num, book_size, bos_token_id)

    samples = []
    for req_id, enc_path in enumerate(encoder_paths):
        encoder_feats = np.load(enc_path)
        # We can't reconstruct the original text tokens from the features, so
        # we leave input_ids empty.  Down-stream pipelines that rely on the
        # encoder features should load them from *encoder_features_path*.
        sample = {
            "id": req_id,
            "input_ids": [],  # Placeholder – not used by decoder-only model
            "context_ids": context_ids,
            "output_len": output_len,
            "task_id": -1,
            "encoder_features_path": os.path.abspath(enc_path),
        }
        samples.append(sample)

    metadata = {
        "workload_type": "t5tts-real",
        "num_requests": len(samples),
        "books_num": books_num,
        "book_size": book_size,
        "bos_token_id": bos_token_id,
        "output_len": output_len,
        "decoder_context_path": os.path.abspath(decoder_context_path),
        "encoder_glob": encoder_glob,
    }

    json_payload = {"metadata": metadata, "samples": samples}

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Wrote {len(samples)} samples to {output_file}")


def cli():
    parser = argparse.ArgumentParser(description="Generate a benchmark JSON using real T5-TTS request data.")
    parser.add_argument("--encoder_glob", type=str,
                        default="/code/tensorrt_llm/encoder_outputs/encoder_outputs_*.npy",
                        help="Glob pattern for encoder output *.npy files.")
    parser.add_argument("--decoder_context", type=str,
                        default="/code/tensorrt_llm/encoder_outputs/decoder_context.npy",
                        help="Path to decoder context *.npy file (before transformations).")
    parser.add_argument("--output", type=str, default="t5tts_real_requests.json",
                        help="Output JSON path.")
    parser.add_argument("--books_num", type=int, default=BOOKS_NUM_DEFAULT)
    parser.add_argument("--book_size", type=int, default=BOOK_SIZE_DEFAULT)
    parser.add_argument("--bos_token_id", type=int, default=BOS_TOKEN_ID_DEFAULT)
    parser.add_argument("--output_len", type=int, default=OUTPUT_LEN_DEFAULT)

    args = parser.parse_args()

    generate_samples(
        encoder_glob=args.encoder_glob,
        decoder_context_path=Path(args.decoder_context),
        output_file=Path(args.output),
        books_num=args.books_num,
        book_size=args.book_size,
        bos_token_id=args.bos_token_id,
        output_len=args.output_len,
    )


if __name__ == "__main__":
    cli() 