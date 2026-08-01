import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tiktoken
from openai import OpenAI, OpenAIError

DEFAULT_MODEL = "gpt-4o"
MAX_TOKENS = 500
TEMPERATURE = 1.0
CHECKPOINT_INTERVAL = 5
RETRY_BASE_DELAY = 1.0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate GPT-4o responses multiple times per prompt with checkpointing"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Excel file with Participant_ID and metadata and placeholders for generated columns")
    parser.add_argument("--prompts-dir", "-p", type=Path, required=True,
                        help="Directory containing prompt .txt files named by Participant_ID")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Final Excel output path")
    parser.add_argument("--api-key", "-k", type=str,
                        default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--repeats", "-r", type=int, default=5,
                        help="Number of responses per prompt (default=5)")
    parser.add_argument("--checkpoint-dir", "-c", type=Path, default=None,
                        help="Directory to save checkpoint Excel files")
    parser.add_argument("--resume", "-x", type=Path, default=None,
                        help="Path to existing checkpoint file to resume from")
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def list_prompt_files(prompts_dir: Path) -> List[Path]:
    return sorted(prompts_dir.glob("*.txt"))


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def create_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_gpt4o_with_retry(
    client: OpenAI,
    system_text: str,
    user_text: str,
    retries: int = 3
) -> tuple[str, str]:
    ending_text = "Remember to first provide your rationale analysis, and only at the very end state your diagnosis."
    additional_text = "Following examples 1 and 2, please create a similar rationale and diagnosis for example 3."

    final_prompt = system_text
    if ending_text in system_text:
        parts = system_text.split(ending_text, 1)
        final_prompt = parts[0] + ending_text + "\n" + additional_text + parts[1]

    delay = RETRY_BASE_DELAY
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": final_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return resp.choices[0].message.content, final_prompt
        except OpenAIError as e:
            logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
    logging.error("All retry attempts failed; returning empty string")
    return "", final_prompt


def checkpoint_save(df: pd.DataFrame, ckpt_dir: Path, count_ids: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint_{count_ids:04d}.xlsx"
    df.to_excel(ckpt_dir / filename, index=False)
    logging.info(f"Saved checkpoint: {filename}")


def extract_prompt_preview(text: str) -> str:
    if len(text) <= 1000:
        return text

    first_500 = text[:500]
    last_500 = text[-10000:]

    return f"{first_500}\n...[중략]...\n{last_500}"


def process_all(
    df: pd.DataFrame,
    prompt_paths: List[Path],
    client: OpenAI,
    repeats: int,
    checkpoint_dir: Optional[Path]
) -> pd.DataFrame:
    for i in range(1, repeats + 1):
        col_r = f"Generated_rationale_{i}"
        col_t = f"Generated_token_{i}"

        if col_r not in df.columns:
            df[col_r] = ""
        if col_t not in df.columns:
            df[col_t] = 0

    if "Prompt_preview" not in df.columns:
        df["Prompt_preview"] = ""

    total_ids = len(prompt_paths)
    total_expected = total_ids * repeats
    existing = sum(
        df[f"Generated_rationale_{i}"].astype(bool).sum()
        for i in range(1, repeats + 1)
    )
    logging.info(f"Resuming: {existing}/{total_expected} responses already generated")

    start_time = time.time()
    processed = 0

    for idx, path in enumerate(prompt_paths, start=1):
        pid = path.stem
        row_idx = df.index[df['Participant_ID'].astype(str) == pid]
        if len(row_idx) != 1:
            logging.warning(f"Participant_ID {pid} mismatch, skipping")
            continue
        r = row_idx[0]
        system_text = path.read_text(encoding='utf-8')

        for i in range(1, repeats + 1):
            col_r, col_t = f"Generated_rationale_{i}", f"Generated_token_{i}"
            if df.at[r, col_r]:
                continue
            response, final_prompt = call_gpt4o_with_retry(client, system_text, "Medical Rationale: ")

            df.at[r, "Prompt_preview"] = extract_prompt_preview(final_prompt)

            df.at[r, col_r] = response
            df.at[r, col_t] = count_tokens(response)
        processed += 1

        if checkpoint_dir and processed % CHECKPOINT_INTERVAL == 0:
            checkpoint_save(df, checkpoint_dir, processed)

        elapsed = time.time() - start_time
        avg = elapsed / idx
        remaining = avg * (total_ids - idx)
        generated = sum(
            df[f"Generated_rationale_{i}"].astype(bool).sum()
            for i in range(1, repeats + 1)
        )
        logging.info(
            f"Processed {idx}/{total_ids} IDs ({processed} new). "
            f"Responses: {generated}/{total_expected}. "
            f"Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s"
        )

    return df


def main() -> None:
    args = parse_args()
    setup_logging()
    client = create_openai_client(args.api_key)
    df = load_dataframe(args.resume or args.input)
    prompts = list_prompt_files(args.prompts_dir)
    updated = process_all(df, prompts, client, args.repeats, args.checkpoint_dir)
    updated.to_excel(args.output, index=False)
    logging.info(f"All done. Results saved to {args.output}")


if __name__ == '__main__':
    INPUT_FILE = Path('data/exp6/experiment_6_test_with_shots_with_tokens.xlsx')
    PROMPTS_DIR = Path('data/prompts_exp6_test')
    OUTPUT_FILE = Path('data/exp6/experiment_6_test_final.xlsx')
    API_KEY = os.getenv('OPENAI_API_KEY')
    REPEATS = 1
    CHECKPOINT_DIR = Path('data/checkpoints')
    RESUME_FILE: Optional[Path] = None

    import sys
    sys.argv = [
        sys.argv[0],
        '--input', str(INPUT_FILE),
        '--prompts-dir', str(PROMPTS_DIR),
        '--output', str(OUTPUT_FILE),
        '--api-key', API_KEY,
        '--repeats', str(REPEATS),
        '--checkpoint-dir', str(CHECKPOINT_DIR),
    ]
    if RESUME_FILE:
        sys.argv += ['--resume', str(RESUME_FILE)]
    main()
