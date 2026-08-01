from pathlib import Path
import re
import argparse
import pandas as pd

BASE_DIR = Path('data/daic_woz/gpt4o_llm_selection/')
SHARE_DIR = Path('')

FILE_NAME = 'gpt4o_llm_exp6_train.xlsx'

DEFAULT_INPUT_PATH = BASE_DIR / FILE_NAME
DEFAULT_OUTPUT_PATH = BASE_DIR / SHARE_DIR / FILE_NAME

def parse_execution_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Selected_Rationale column and add modified_rationale column"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the input Excel file (default: {DEFAULT_INPUT_PATH})"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the cleaned output Excel file (default: {DEFAULT_OUTPUT_PATH})"
    )
    return parser.parse_args()


def load_excel_file(source_file_path: Path) -> pd.DataFrame:
    return pd.read_excel(source_file_path)


def clean_rationale_text(original_text: str) -> str:
    if not isinstance(original_text, str):
        return ""
    without_bold = re.sub(r"\*\*", "", original_text)
    cleaned = re.sub(r"Diagnosis:.*$", "", without_bold, flags=re.IGNORECASE)
    return cleaned.strip()


def add_cleaned_rationale_column(
    dataframe: pd.DataFrame,
    source_column_name: str,
    target_column_name: str
) -> pd.DataFrame:
    dataframe[target_column_name] = dataframe[source_column_name].apply(clean_rationale_text)
    return dataframe


def save_dataframe_to_excel(dataframe: pd.DataFrame, destination_file_path: Path) -> None:
    dataframe.to_excel(destination_file_path, index=False)


def main() -> None:
    config = parse_execution_config()
    input_path = config.input
    output_path = config.output

    df = load_excel_file(input_path)
    df = add_cleaned_rationale_column(df, "Selected_Rationale", "modified_rationale")
    save_dataframe_to_excel(df, output_path)
    print(f"Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    main()
