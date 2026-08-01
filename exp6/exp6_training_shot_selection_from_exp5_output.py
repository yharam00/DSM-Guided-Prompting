import pandas as pd
import random
from pathlib import Path
from typing import Tuple

def load_dataset(filepath: Path) -> pd.DataFrame:
    return pd.read_excel(filepath)

def create_unique_names(
    df: pd.DataFrame,
    id_column: str,
    rationale_num_column: str,
    output_column: str
) -> None:
    df[output_column] = (
        df[id_column].astype(str)
        + "_"
        + df[rationale_num_column].astype(str)
    )

def select_random_shot_for_row(
    row: pd.Series,
    candidates: pd.DataFrame,
    id_column: str,
    name_column: str
) -> str:
    filtered = candidates[candidates[id_column] != row[id_column]]
    if filtered.empty:
        return ""
    rand_seed = random.randint(0, 2**32 - 1)
    return filtered[name_column].sample(n=1, random_state=rand_seed).iloc[0]

def assign_shots(
    df: pd.DataFrame,
    binary_column: str,
    zero_shot_column: str,
    one_shot_column: str,
    id_column: str,
    name_column: str
) -> None:
    df[zero_shot_column] = ""
    df[one_shot_column] = ""
    group_zero = df[df[binary_column] == 0]
    group_one  = df[df[binary_column] == 1]

    for idx, row in df.iterrows():
        df.at[idx, zero_shot_column] = select_random_shot_for_row(
            row, group_zero, id_column, name_column
        )
        df.at[idx, one_shot_column] = select_random_shot_for_row(
            row, group_one, id_column, name_column
        )

def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    df.to_excel(output_path, index=False)

def main() -> None:

    random.seed(42)

    input_path  = Path("data/exp6/experiment_5_train_selected.xlsx")
    output_path = Path("data/exp6/experiment_5_train_selected_with_shots.xlsx")

    df = load_dataset(input_path)
    create_unique_names(
        df,
        id_column="Participant_ID",
        rationale_num_column="Exp_5_Selected_Rationale_Number",
        output_column="name"
    )
    assign_shots(
        df,
        binary_column="PHQ_Binary",
        zero_shot_column="zero_shot_name",
        one_shot_column="one_shot_name",
        id_column="Participant_ID",
        name_column="name"
    )
    save_dataset(df, output_path)

if __name__ == "__main__":
    main()
