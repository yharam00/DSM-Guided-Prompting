import pandas as pd
import tiktoken
from pathlib import Path
from typing import Dict, Any


def load_data(excel_path: Path) -> pd.DataFrame:
    return pd.read_excel(excel_path)


def build_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return df.set_index("name").to_dict(orient="index")


def format_prompt(
    row: Dict[str, Any],
    lookup: Dict[str, Dict[str, Any]],
    dsm5_criteria: str
) -> str:
    status = "depressed" if row["PHQ_Binary"] == 1 else "non-depressed"

    def get_shot_fields(shot_key: str):
        shot_name = row[shot_key]
        shot = lookup[shot_name]
        return {
            "gender": shot["Gender"],
            "transcript": shot["Transcript"],
            "rationale": shot["Exp_5_Selected_Rationale"]
        }

    zero = get_shot_fields("zero_shot_name")
    one = get_shot_fields("one_shot_name")
    self_data = {"gender": row["Gender"], "transcript": row["Transcript"]}

    template = f"""
Generate detailed medical rationale for the diagnosis ("Diagnosis: {status}") based on the patient description.
These rationales should be the crucial cue for the diagnosis. Pretend that you don't know the diagnosis ("Diagnosis: {status}").

Example 1
Patient Description: Here is a transcript of the {zero['gender']} patient's clinical interview, containing only the patient's utterances.
{zero['transcript']}
Also, based on DSM-5 diagnostic criteria for a major depressive episode.
{dsm5_criteria}
Diagnosis: non-depressed
Medical Rationale:
{zero['rationale']}

Example 2
Patient Description: Here is a transcript of the {one['gender']} patient's clinical interview, containing only the patient's utterances.
{one['transcript']}
Also, based on DSM-5 diagnostic criteria for a major depressive episode.
{dsm5_criteria}
Diagnosis: depressed
Medical Rationale:
{one['rationale']}

Example 3
Patient Description: Here is a transcript of the {self_data['gender']} patient's clinical interview, containing only the patient's utterances.
{self_data['transcript']}
Also, based on DSM-5 diagnostic criteria for a major depressive episode.
{dsm5_criteria}
Diagnosis: {status}
Be concise but thorough. Keep your response under 500 tokens (about 375 words).
""".strip()
    return template


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def save_prompt_file(output_dir: Path, name: str, prompt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{name}.txt"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(prompt)


def main():
    excel_path = Path('data/exp6/experiment_5_train_selected_with_shots.xlsx')
    output_dir = Path('data/prompts_exp6_training')
    dsm5_criteria = (
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

    df = load_data(excel_path)
    lookup = build_lookup(df)

    token_counts = {}

    for _, row in df.iterrows():
        name = row["name"]
        prompt = format_prompt(row.to_dict(), lookup, dsm5_criteria)
        tokens = count_tokens(prompt)

        token_counts[name] = tokens

        save_prompt_file(output_dir, name, prompt)

    df['Token_Count'] = df['name'].map(token_counts)

    output_excel = excel_path.parent / 'experiment_5_train_selected_with_shots_with_tokens.xlsx'
    df.to_excel(output_excel, index=False)
    print(f"엑셀 파일이 저장되었습니다: {output_excel}")


if __name__ == "__main__":
    main()
