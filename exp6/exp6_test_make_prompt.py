import pandas as pd
import tiktoken
from pathlib import Path

DSM5_CRITERIA = (
    """
A. Five (or more) of the following symptoms have been present during the same 2-week period and represent a change from previous functioning; at least one of the symptoms is either (1) depressed mood or (2) loss of interest or pleasure.
*Note: Do not include symptoms that are clearly attributable to another medical condition.
1. Depressed mood most of the day, nearly every day, as indicated by either subjective report(e.g., feels sad, empty, hopeless) or observation made by others (e.g., appears tearful).
2. Markedly diminished interest or pleasure in all, or almost all, activities most of the day, nearly every day (as indicated by subjective account or observation).
3. Significant weight loss when not dieting or weight gain (e.g., change of more than 5% of body weight in a month), or decrease or increase in appetite nearly every day.
4. Insomnia or hypersomnia nearly every day.
5. Psychomotor agitation or retardation nearly every day (observable by others, not merely subjective feelings of restlessness or being slowed down).
6. Fatigue or loss of energy nearly every day.
7. Feelings of worthlessness or excessive or inappropriate guilt (which may be delusional) nearly every day (not merely self-reproach or guilt about being sick).
8. Diminished ability to think or concentrate, or indecisiveness, nearly every day (either by subjective account or as observed by others).
9. Recurrent thoughts of death (not just fear of dying), recurrent suicidal ideation without a specific plan, or a suicide attempt or a specific plan for committing suicide.
B. The symptoms cause clinically significant distress or impairment in social, occupational, or other important areas of functioning.
C. The episode is not attributable to the physiological effects of a substance or to another medical condition.
* Note: Criteria A through C represent a major depressive episode.
D. The occurrence of the major depressive episode is not better explained by schizoaffective disorder, schizophrenia, schizophreniform disorder, delusional disorder, or other specified and unspecified schizophrenia spectrum and other psychotic disorders.
E. There has never been a manic episode or a hypomanic episode.
* Note: This exclusion does not apply if all of the manic-like or hypomanic-like episodes are substance-induced or are attributable to the physiological effects of another medical condition.
    """
)

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)

def build_train_lookup(train_df: pd.DataFrame) -> dict:
    return train_df.set_index("name")[["Gender", "Transcript", "Generated_Rationale"]].to_dict(orient="index")

def build_prompt(row: pd.Series, lookup: dict) -> str:
    def get_fields(shot_key: str):
        shot_name = row[shot_key]
        if pd.notna(shot_name) and shot_name in lookup:
            data = lookup[shot_name]
        else:
            data = {
                "Gender": row["Gender"],
                "Transcript": row["Transcript"],
                "Generated_Rationale": row["Generated_Rationale"]
            }
        return data["Gender"], data["Transcript"], data["Generated_Rationale"]

    gender_zero, transcript_zero, rationale_zero = get_fields("zero_shot_name")
    gender_one,  transcript_one,  rationale_one  = get_fields("one_shot_name")
    gender_self   = row["Gender"]
    transcript_self = row["Transcript"]

    prompt = f"""You are a doctor. Generate medical rationale and diagnose the patient based on the information your colleague gave you.
You can utilize the medical rationale.
Answer me with only either "depressed" or "non-depressed". You should follow the style of the history

Example 1
Patient Description: Here is a transcript of the {gender_zero} patient's clinical interview, containing only the patient's utterances.
{transcript_zero}
Also, based on DSM-5 diagnostic criteria for a major depressive episode.
{DSM5_CRITERIA}
Medical Rationale:
{rationale_zero}
Diagnosis: non-depressed

Example 2
Patient Description: Here is a transcript of the {gender_one} patient's clinical interview, containing only the patient's utterances.
{transcript_one}
Also, based on DSM-5 diagnostic criteria for a major depressive episode.
{DSM5_CRITERIA}
Medical Rationale:
{rationale_one}
Diagnosis: depressed

Example 3
Patient Description: Here is a transcript of the {gender_self} patient's clinical interview, containing only the patient's utterances.
{transcript_self}
Also, based on DSM-5 diagnostic criteria for a major depressive episode.
{DSM5_CRITERIA}
Remember to first provide your rationale analysis, and only at the very end state your diagnosis.
Be concise but thorough. Keep your response under 500 tokens (about 375 words).
"""
    return prompt

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def main():
    TEST_FILE    = Path('data/exp6/experiment_6_test_with_shots.xlsx')
    TRAIN_FILE   = Path('data/exp6/experiment_6_train_final.xlsx')
    OUTPUT_FILE  = Path("data/exp6/experiment_6_test_with_shots_with_tokens.xlsx")
    PROMPTS_DIR  = Path('data/prompts_exp6_test')
    MODEL_NAME   = "gpt-3.5-turbo"

    test_df = load_data(TEST_FILE)
    train_df = load_data(TRAIN_FILE)
    lookup = build_train_lookup(train_df)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    token_counts = []

    for _, row in test_df.iterrows():
        pid = row["Participant_ID"]
        prompt_text = build_prompt(row, lookup)

        prompt_path = PROMPTS_DIR / f"{pid}.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        tokens = count_tokens(prompt_text, MODEL_NAME)
        token_counts.append(tokens)

    test_df["Token_Count"] = token_counts
    test_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved prompts in {PROMPTS_DIR} and updated Excel to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
