import pandas as pd
import os
from typing import Optional

def create_prompt_column(input_file: str, output_file: Optional[str] = None) -> None:

    dsm5_criteria = """A. Five (or more) of the following symptoms have been present during the same 2-week period and represent a change from previous functioning; at least one of the symptoms is either (1) depressed mood or (2) loss of interest or pleasure.
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
* Note: This exclusion does not apply if all of the manic-like or hypomanic-like episodes are substance-induced or are attributable to the physiological effects of another medical condition."""

    prompt_template = (
        "Generate detailed medical rationale for the diagnosis (\"Diagnosis:\" {depression_status}) "
        "based on the patient description. These rationales should be the crucial cue for the diagnosis. "
        "Pretend that you do not know the diagnosis.\n"
        "Patient Description: Here is a transcript of the {gender} patient's clinical interview, "
        "containing only the patient's utterances.\n{transcript}\n"
        "Also, based on DSM-5 diagnostic criteria for a major depressive episode.\n{dsm5_criteria}\n"
        "Diagnosis: {depression_status}\nBe concise but thorough. Keep your response under 500 tokens (about 375 words)."
    )

    try:
        print(f"엑셀 파일을 읽는 중: {input_file}")
        df = pd.read_excel(input_file)

        required_columns = ['Gender', 'PHQ_Binary', 'Transcript']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"필요한 열이 없습니다: {missing_columns}")

        print(f"데이터 행 수: {len(df)}")
        print(f"기존 열: {list(df.columns)}")

        def get_depression_status(phq_binary):
            if pd.isna(phq_binary):
                return "unknown"
            return "depressed" if phq_binary == 1 else "non-depressed"

        print("Prompt 열을 생성하는 중...")
        prompts = []

        for index, row in df.iterrows():
            depression_status = get_depression_status(row['PHQ_Binary'])
            gender = row['Gender'] if not pd.isna(row['Gender']) else "patient"
            transcript = row['Transcript'] if not pd.isna(row['Transcript']) else ""

            prompt = prompt_template.format(
                depression_status=depression_status,
                gender=gender,
                transcript=transcript,
                dsm5_criteria=dsm5_criteria
            )

            prompts.append(prompt)

            if (index + 1) % 10 == 0:
                print(f"처리 완료: {index + 1}/{len(df)}")

        df['Prompt'] = prompts

        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_with_prompt.xlsx"

        print(f"결과를 저장하는 중: {output_file}")
        df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"완료! 새로운 파일이 저장되었습니다: {output_file}")
        print(f"새로운 열: {list(df.columns)}")

        if len(df) > 0:
            print("\n--- 첫 번째 프롬프트 미리보기 ---")
            print(df['Prompt'].iloc[0][:200] + "..." if len(df['Prompt'].iloc[0]) > 200 else df['Prompt'].iloc[0])

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {input_file}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")

def main():
    input_file = 'data/results/30_rational_candidate_deepseek_v3_experiment_5.xlsx'

    if not os.path.exists(input_file):
        print(f"파일이 존재하지 않습니다: {input_file}")
        return

    output_file = 'data/results/30_rational_candidate_deepseek_v3_experiment_5.xlsx'
    if not output_file:
        output_file = None

    create_prompt_column(input_file, output_file)

if __name__ == "__main__":
    main()
