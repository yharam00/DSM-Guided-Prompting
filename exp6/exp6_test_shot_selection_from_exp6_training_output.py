import pandas as pd
import random
from pathlib import Path

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def save_data(df: pd.DataFrame, path: Path) -> None:
    df.to_excel(path, index=False)

def assign_unique_shots(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    binary_col: str = "PHQ_Binary",
    name_col: str = "name",
    zero_out: str = "zero_shot_name",
    one_out: str = "one_shot_name",
    seed: int = 20250508
) -> pd.DataFrame:
    rng = random.Random(seed)

    zero_pool = train_df[train_df[binary_col] == 0][name_col].tolist()
    one_pool  = train_df[train_df[binary_col] == 1][name_col].tolist()

    n = len(test_df)
    if len(zero_pool) < n or len(one_pool) < n:
        raise ValueError(
            f"Not enough samples: need {n} zeros and {n} ones, "
            f"but got {len(zero_pool)} zero-pool and {len(one_pool)} one-pool."
        )

    zero_choices = rng.sample(zero_pool, n)
    one_choices  = rng.sample(one_pool,  n)

    test_df[zero_out] = zero_choices
    test_df[one_out]  = one_choices

    return test_df

def main():
    TRAIN_PATH = Path("data/exp6/experiment_6_train_final.xlsx")
    TEST_PATH  = Path("data/results/e_daic_transcript_test_modified_20250507_084908.xlsx")
    OUTPUT_PATH = Path("data/exp6/experiment_6_test_with_shots.xlsx")
    SEED = 20250508

    train_df = load_data(TRAIN_PATH)
    test_df  = load_data(TEST_PATH)

    result_df = assign_unique_shots(
        train_df,
        test_df,
        binary_col="PHQ_Binary",
        name_col="name",
        zero_out="zero_shot_name",
        one_out="one_shot_name",
        seed=SEED
    )

    save_data(result_df, OUTPUT_PATH)
    print(f"Saved new test file with shots to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
