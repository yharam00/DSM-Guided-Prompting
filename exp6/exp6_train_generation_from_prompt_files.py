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
USER_PROMPT = "Medical Rationale: "
MAX_TOKENS = 500
TEMPERATURE = 1.0
CHECKPOINT_INTERVAL = 10
RETRY_BASE_DELAY = 1.0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate GPT-4o medical rationales with checkpointing"
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Path to the Excel file with prompt names and metadata")
    parser.add_argument("--prompts-dir", "-p", type=Path, required=True,
                        help="Directory containing .txt prompt files")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Final Excel output path")
    parser.add_argument("--api-key", "-k", type=str,
                        default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--checkpoint-dir", "-c", type=Path, default=None,
                        help="Directory to write checkpoint Excel files")
    parser.add_argument("--resume", "-r", type=Path, default=None,
                        help="Path to existing checkpoint to resume from")
    return parser.parse_args()


def load_dataframe(excel_path: Path) -> pd.DataFrame:
    return pd.read_excel(excel_path)


def list_prompt_files(prompts_dir: Path) -> List[Path]:
    return sorted(prompts_dir.glob('*.txt'))


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))


def create_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def call_gpt4o_with_retry(
    client: OpenAI,
    system_content: str,
    user_content: str,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int = 3,
) -> str:
    delay = RETRY_BASE_DELAY
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except OpenAIError as err:
            logging.warning(
                f"Attempt {attempt}/{retries} failed: {err}. Retrying in {delay}s"
            )
            time.sleep(delay)
            delay *= 2
    logging.error("All retry attempts failed. Returning empty string.")
    return ""


def checkpoint_save(
    df: pd.DataFrame,
    checkpoint_dir: Path,
    count: int
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint_{count:04d}.xlsx"
    path = checkpoint_dir / filename
    df.to_excel(path, index=False)
    logging.info(f"Saved checkpoint: {filename}")


def process_all(
    df: pd.DataFrame,
    prompt_paths: List[Path],
    client: OpenAI,
    model: str,
    user_prompt: str,
    checkpoint_dir: Optional[Path],
) -> pd.DataFrame:
    if 'Generated_Rationale' not in df.columns:
        df['Generated_Rationale'] = ""
        df['Generated_Token_Count'] = 0
    else:
        df['Generated_Rationale'] = df['Generated_Rationale'].fillna("")
        df['Generated_Token_Count'] = df['Generated_Token_Count'].fillna(0).astype(int)

    total = len(prompt_paths)
    start_time = time.time()
    processed_new = 0

    for idx, file_path in enumerate(prompt_paths, start=1):
        name = file_path.stem
        existing = df.loc[df['name'] == name, 'Generated_Rationale'].iloc[0]
        if existing:
            logging.info(f"Skipping already processed ({idx}/{total}): {name}")
            continue

        system_text = file_path.read_text(encoding='utf-8')
        rationale = call_gpt4o_with_retry(
            client,
            system_text,
            USER_PROMPT,
            model,
            MAX_TOKENS,
            TEMPERATURE,
        )
        token_count = count_tokens(rationale)

        df.loc[df['name'] == name, ['Generated_Rationale', 'Generated_Token_Count']] = [
            rationale,
            token_count,
        ]
        processed_new += 1

        if checkpoint_dir and processed_new % CHECKPOINT_INTERVAL == 0:
            checkpoint_save(df, checkpoint_dir, processed_new)

        elapsed = time.time() - start_time
        avg = elapsed / idx
        remaining = avg * (total - idx)
        logging.info(
            f"Progress: {idx}/{total} files ({processed_new} new). "
            f"Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s"
        )

    return df


def main() -> None:
    args = parse_args()
    setup_logging()

    client = create_openai_client(args.api_key)
    df = load_dataframe(args.resume) if args.resume else load_dataframe(args.input)
    prompt_files = list_prompt_files(args.prompts_dir)

    updated_df = process_all(
        df,
        prompt_files,
        client,
        DEFAULT_MODEL,
        USER_PROMPT,
        args.checkpoint_dir,
    )
    updated_df.to_excel(args.output, index=False)
    logging.info(f"Completed. Results saved to {args.output}")


if __name__ == '__main__':
    INPUT = Path('data/exp6/experiment_5_train_selected_with_shots_with_tokens.xlsx')
    PROMPTS_DIR = Path('data/prompts_exp6_training')
    OUTPUT = Path('data/exp6/experiment_6_train_final.xlsx')
    API_KEY = os.getenv('OPENAI_API_KEY2') or os.getenv('OPENAI_API_KEY')
    CHECKPOINT_DIR = Path('data/checkpoints')
    RESUME_FILE: Optional[Path] = None

    import sys
    cli_args = [
        sys.argv[0],
        '--input', str(INPUT),
        '--prompts-dir', str(PROMPTS_DIR),
        '--output', str(OUTPUT),
        '--api-key', API_KEY,
        '--checkpoint-dir', str(CHECKPOINT_DIR),
    ]
    if RESUME_FILE:
        cli_args += ['--resume', str(RESUME_FILE)]
    sys.argv = cli_args
    main()
