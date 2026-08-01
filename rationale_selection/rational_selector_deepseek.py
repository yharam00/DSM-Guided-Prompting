import pandas as pd
from openai import OpenAI
import os
import re
from typing import List, Dict, Any
import json
import logging
from dotenv import load_dotenv

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RationalSelector:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            api_key=api_key or os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    def read_excel_file(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(file_path)
            logger.info(f"엑셀 파일을 성공적으로 읽었습니다: {file_path}")
            logger.info(f"데이터 형태: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"엑셀 파일 읽기 실패: {e}")
            raise

    def find_rational_columns(self, df: pd.DataFrame) -> List[str]:
        rational_columns = []
        pattern = re.compile(r'^Rational_\d+$')

        for col in df.columns:
            if pattern.match(str(col)):
                rational_columns.append(col)

        rational_columns.sort(key=lambda x: int(x.split('_')[1]))
        logger.info(f"발견된 Rational 열들: {rational_columns}")
        return rational_columns

    def check_prompt_column(self, df: pd.DataFrame) -> bool:
        prompt_exists = 'Prompt' in df.columns
        logger.info(f"Prompt 열 존재: {prompt_exists}")
        return prompt_exists

    def get_rational_selection_prompt(self, rationals: List[str], prompt_content: str, n_select: int) -> str:
        rational_sections = []
        for i, rational in enumerate(rationals):
            rational_section = f"""
--- RATIONALE {i+1} ---
{rational}
"""
            rational_sections.append(rational_section)

        rational_text = "\n".join(rational_sections)

        prompt = f"""
You are a medical expert. Your task is to evaluate and select the best rationale(s) based on medical accuracy and clinical reliability.

--- ORIGINAL PROMPT ---
{prompt_content}

--- GENERATED RATIONALES ---
{rational_text}

--- TASK ---
Select the best {n_select} rationale(s) from the {len(rationals)} options above.

Evaluation criteria:
1. Medical Accuracy
2. Clinical Reliability
3. Evidence-based Reasoning

IMPORTANT: These rationales were generated specifically for the medical case shown above.

--- RESPONSE FORMAT ---
Respond with ONLY the rationale number(s) in JSON array format.

Your answer:
"""
        return prompt

    def call_gpt4o_api(self, prompt: str) -> List[int]:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a medical expert evaluating clinical rationales. You must select the most medically accurate and clinically reliable rationales from the given options."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=100
            )

            response_text = response.choices[0].message.content.strip()
            logger.info(f"GPT-4O 응답: {response_text}")

            try:
                json_match = re.search(r'\[[\d,\s]+\]', response_text)
                if json_match:
                    json_str = json_match.group()
                    selected_numbers = json.loads(json_str)
                else:
                    numbers = re.findall(r'\d+', response_text)
                    selected_numbers = [int(num) for num in numbers]

                logger.info(f"선택된 rational 번호들: {selected_numbers}")
                return selected_numbers

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"JSON 파싱 실패: {e}")
                numbers = re.findall(r'\d+', response_text)
                return [int(num) for num in numbers]

        except Exception as e:
            logger.error(f"GPT-4O API 호출 실패: {e}")
            raise

    def process_excel_file(self, input_path: str, output_path: str):
        df = self.read_excel_file(input_path)

        rational_columns = self.find_rational_columns(df)

        if not rational_columns:
            raise ValueError("Rational 열을 찾을 수 없습니다.")

        has_prompt_column = self.check_prompt_column(df)
        if not has_prompt_column:
            logger.warning("Prompt 열을 찾을 수 없습니다. 빈 프롬프트로 진행합니다.")

        n_rationals = len(rational_columns)
        logger.info(f"총 {n_rationals}개의 Rational 열 발견")

        if n_rationals == 30:
            n_select = 5
        elif n_rationals == 15:
            n_select = 5
        elif n_rationals == 5:
            n_select = 1
        else:
            logger.warning(f"예상하지 못한 Rational 개수: {n_rationals}. 기본값 1개 선택으로 설정.")
            n_select = 1

        logger.info(f"{n_rationals}개 중 {n_select}개 선택 예정")

        new_columns = {}
        for i in range(n_select):
            new_columns[f'Selected_Rational_Number_{i+1}'] = []

        for index, row in df.iterrows():
            logger.info(f"행 {index + 1}/{len(df)} 처리 중...")

            rationals = []
            for col in rational_columns:
                rational_text = str(row[col]) if pd.notna(row[col]) else ""
                rationals.append(rational_text)

            prompt_content = ""
            if has_prompt_column:
                prompt_content = str(row['Prompt']) if pd.notna(row['Prompt']) else "No specific prompt provided."
            else:
                prompt_content = "No prompt column found in the data."

            prompt = self.get_rational_selection_prompt(rationals, prompt_content, n_select)
            selected_numbers = self.call_gpt4o_api(prompt)

            for i in range(n_select):
                if i < len(selected_numbers):
                    new_columns[f'Selected_Rational_Number_{i+1}'].append(selected_numbers[i])
                else:
                    new_columns[f'Selected_Rational_Number_{i+1}'].append(None)

        for col_name, values in new_columns.items():
            df[col_name] = values

        df.to_excel(output_path, index=False)
        logger.info(f"결과 파일 저장 완료: {output_path}")

        return df

def main():
    selector = RationalSelector(api_key=DEEPSEEK_API_KEY)

    input_file = 'data/results/e_daic_transcript_experiment_2_id300-486_deepseek-chat_20251003_213344.xlsx'
    output_file = 'data/results/e_daic_transcript_experiment_2_depression_deepseek_generated_deepseek_selected.xlsx'

    try:
        result_df = selector.process_excel_file(input_file, output_file)
        print("처리가 완료되었습니다!")
        print(f"결과 파일: {output_file}")
        print(f"최종 데이터 형태: {result_df.shape}")

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        print(f"오류: {e}")

if __name__ == "__main__":
    main()
