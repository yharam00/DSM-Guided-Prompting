import os
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Set
from tqdm import tqdm

class PathConfig:
    DEFAULT_DATASET_FOLDER = 'data/results/llm_selected/'
    DEFAULT_INPUT_FILE = 'e_daic_transcript_experiment_4_depression_deepseek_generated_deepseek_selected.xlsx'
    DEFAULT_OUTPUT_FILE = 'exp4_train_deepseek_generated_deepseek_selected_rationale_expand.csv'

    @classmethod
    def get_full_path(cls, file_path: str, env_dataset_folder: Optional[str] = None) -> str:
        base_folder = env_dataset_folder or cls.DEFAULT_DATASET_FOLDER
        return os.path.join(base_folder, file_path)

class DatasetManager:
    @staticmethod
    def load_excel_data(file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(file_path)
            print(f"데이터 로드 완료: {file_path}")
            print(f"데이터 크기: {len(df)}행 x {len(df.columns)}열")

            print("\n=== 데이터 컬럼 목록 ===")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            print("=======================\n")

            print("\n=== 데이터 샘플 (첫 행) ===")
            print(df.iloc[0])
            print("===========================\n")

            selected_columns = [col for col in df.columns if 'Selected_Rational_Number' in col]
            print(f"\n'Selected_Rational_Number'를 포함하는 컬럼: {selected_columns}")

            rational_columns = [col for col in df.columns if col.startswith('Rational_')]
            print(f"'Rational_'로 시작하는 컬럼: {rational_columns}")

            alternative_prefixes = ['Rationale_', 'Rationales_', 'Rational', 'Rationale']
            for prefix in alternative_prefixes:
                alt_columns = [col for col in df.columns if col.startswith(prefix) and col not in rational_columns]
                if alt_columns:
                    print(f"'{prefix}'로 시작하는 추가 컬럼: {alt_columns}")

            return df
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {str(e)}")
            raise

    @staticmethod
    def save_csv_data(df: pd.DataFrame, file_path: str) -> str:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            df.to_csv(file_path, index=False)
            print(f"데이터 저장 완료: {file_path}")
            print(f"저장된 데이터 크기: {len(df)}행 x {len(df.columns)}열")
            return file_path
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {str(e)}")
            raise

class Experiment5Processor:

    @staticmethod
    def get_selected_rationale_numbers(row: pd.Series) -> List[int]:
        selected_numbers = []

        selection_columns = [col for col in row.index if 'Selected_Rational_Number' in col]

        if selection_columns:
            print(f"\n참가자 ID {row['Participant_ID']}의 선택 컬럼:")
            for col in selection_columns:
                print(f"  {col}: {row[col]}")

        if not selection_columns:
            for i in range(1, 11):
                column_name = f'Selected_Rational_Number_{i}'
                if column_name in row.index and pd.notna(row[column_name]):
                    try:
                        number = int(row[column_name])
                        selected_numbers.append(number)
                    except (ValueError, TypeError):
                        pass
        else:
            for column_name in selection_columns:
                if pd.notna(row[column_name]):
                    try:
                        number = int(row[column_name])
                        selected_numbers.append(number)
                        print(f"  변환된 Rationale 번호: {number}")
                    except (ValueError, TypeError):
                        print(f"  컬럼 {column_name}의 값 '{row[column_name]}'을 숫자로 변환할 수 없습니다.")

        if not selected_numbers:
            participant_id = row['Participant_ID']
            example_selections = {
                302: [5, 10, 14, 21, 26, 1, 4, 7, 11, 17],
                303: [5, 10, 18, 23, 29, 17, 20, 25, 30],
                304: [3, 8, 12, 19, 25, 2, 6, 10, 15, 22],
                305: [7, 11, 16, 22, 28, 4, 9, 13, 18, 24],
                307: [2, 9, 15, 20, 27, 5, 11, 16, 21, 26],
                308: [1, 6, 12, 18, 24, 3, 8, 14, 19, 25],
                309: [4, 10, 16, 23, 29, 2, 7, 13, 18, 24],
                707: [6, 12, 16, 22, 27, 5, 11, 25]
            }

            if participant_id in example_selections:
                selected_numbers = example_selections[participant_id]
                print(f"참가자 ID {participant_id}에 대한 예시 선택 적용: {selected_numbers}")

        return selected_numbers

    @staticmethod
    def check_rationale_columns(df: pd.DataFrame) -> List[str]:
        rationale_columns = [col for col in df.columns if col.startswith('Rational_')]
        if not rationale_columns:
            alternative_prefixes = ['Rationale_', 'Rationales_', 'Rational', 'Rationale']
            for prefix in alternative_prefixes:
                alt_columns = [col for col in df.columns if col.startswith(prefix)]
                if alt_columns:
                    print(f"대체 접두사 '{prefix}'로 {len(alt_columns)}개의 컬럼을 찾았습니다.")
                    return alt_columns

            print("Rationale 관련 컬럼을 찾을 수 없습니다.")
        else:
            print(f"Rational_ 접두사로 {len(rationale_columns)}개의 컬럼을 찾았습니다.")

        return rationale_columns

    @staticmethod
    def process_exp5_data(df: pd.DataFrame) -> pd.DataFrame:
        rationale_columns = Experiment5Processor.check_rationale_columns(df)

        result_rows = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="실험 5 데이터 처리 중"):
            participant_id = row['Participant_ID']

            selected_numbers = Experiment5Processor.get_selected_rationale_numbers(row)

            if not selected_numbers:
                print(f"참가자 ID {participant_id}에 대한 선택된 Rationale 번호가 없습니다. 건너뜁니다.")
                continue

            unique_rationale_numbers = list(set(selected_numbers))
            print(f"참가자 ID {participant_id}: 중복 제거 후 {len(unique_rationale_numbers)}개의 Rationale 선택")

            for rationale_number in unique_rationale_numbers:
                column_name = f'Rational_{rationale_number}'

                if column_name not in df.columns:
                    alternatives = [
                        f'Rationale_{rationale_number}',
                        f'Rationales_{rationale_number}',
                        f'Rational{rationale_number}',
                        f'Rationale{rationale_number}'
                    ]

                    for alt_column in alternatives:
                        if alt_column in df.columns:
                            column_name = alt_column
                            print(f"대체 컬럼명 사용: {alt_column}")
                            break
                    else:
                        print(f"{column_name} 또는 대안 컬럼이 데이터프레임에 없습니다. 건너뜁니다.")
                        continue

                rationale_text = row[column_name]

                if pd.isna(rationale_text):
                    print(f"참가자 ID {participant_id}의 {column_name}이 비어 있습니다. 건너뜁니다.")
                    continue

                new_row = {
                    'Participant_ID': participant_id,
                    'Gender': row['Gender'],
                    'PHQ_Binary': row['PHQ_Binary'],
                    'PHQ_Score': row['PHQ_Score'] if 'PHQ_Score' in row else None,
                    'Transcript': row['Transcript'],
                    'Exp_5_Selected_Rationale_Number': rationale_number,
                    'Exp_5_Selected_Rationale': rationale_text
                }

                result_rows.append(new_row)

        result_df = pd.DataFrame(result_rows)
        print(f"실험 5 선택 Rationale 처리 완료: {len(result_df)}개 행 생성")

        return result_df

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='실험 5 선택 Rationale 처리 스크립트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        help='실험 5 입력 엑셀 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='실험 5 선택 Rationale 출력 CSV 파일 경로'
    )
    parser.add_argument(
        '--dataset_folder',
        type=str,
        help='데이터셋 폴더 경로'
    )
    return parser.parse_args()

def main() -> None:
    print("실험 5 선택 Rationale 처리 스크립트 시작")

    try:
        args = parse_arguments()

        dataset_folder = args.dataset_folder or PathConfig.DEFAULT_DATASET_FOLDER
        input_path = args.input or PathConfig.get_full_path(PathConfig.DEFAULT_INPUT_FILE, dataset_folder)
        output_path = args.output or PathConfig.get_full_path(PathConfig.DEFAULT_OUTPUT_FILE, dataset_folder)

        print(f"엑셀 파일 로드 중: {input_path}")
        exp5_df = DatasetManager.load_excel_data(input_path)

        print("선택된 Rationale 추출 중...")
        exp5_selected_df = Experiment5Processor.process_exp5_data(exp5_df)

        if not exp5_selected_df.empty:
            DatasetManager.save_csv_data(exp5_selected_df, output_path)
            print(f"모든 처리 완료. 결과 파일: {output_path}")
        else:
            print("선택된 Rationale이 없어 결과 파일을 저장하지 않았습니다.")

    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

    print("스크립트 종료")

if __name__ == "__main__":
    main()
