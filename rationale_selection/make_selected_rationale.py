import os
import pandas as pd
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
from tqdm import tqdm
import re

class PathConfig:
    DEFAULT_DATASET_FOLDER = 'data/results/'
    DEFAULT_INPUT_FILE = 'e_daic_transcript_experiment_4_depression_gpt4o_selected.xlsx'
    DEFAULT_OUTPUT_PREFIX = 'e_daic_transcript_experiment_4_gpt4o_expand'

    @classmethod
    def get_full_path(cls, file_path: str, env_dataset_folder: Optional[str] = None) -> str:
        base_folder = env_dataset_folder or cls.DEFAULT_DATASET_FOLDER
        return os.path.join(base_folder, file_path)

    @classmethod
    def get_output_paths(cls, base_path: str) -> Tuple[str, str]:
        base_name = os.path.splitext(base_path)[0]
        csv_path = f"{base_name}.csv"
        excel_path = f"{base_name}.xlsx"
        return csv_path, excel_path

class DatasetManager:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.xlsx' or ext == '.xls':
                df = pd.read_excel(file_path)
                print(f"엑셀 파일 로드 완료: {file_path}")
            elif ext == '.csv':
                df = pd.read_csv(file_path)
                print(f"CSV 파일 로드 완료: {file_path}")
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: .xlsx, .xls, .csv")

            print(f"데이터 크기: {len(df)}행 x {len(df.columns)}열")

            print("\n=== 데이터 컬럼 목록 ===")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            print("=======================\n")

            if not df.empty:
                print("\n=== 데이터 샘플 (첫 행) ===")
                print(df.iloc[0])
                print("===========================\n")

            return df
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {str(e)}")
            raise

    @staticmethod
    def save_data(df: pd.DataFrame, csv_path: str, excel_path: str) -> Tuple[str, str]:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        try:
            df.to_csv(csv_path, index=False)
            print(f"CSV 파일 저장 완료: {csv_path}")

            excel_df = df.copy()
            text_cols = ['Transcript', 'Selected_Rationale']

            excel_limit = 30000
            has_long_texts = False

            for col in text_cols:
                if col in excel_df.columns:
                    long_texts = excel_df[col].apply(
                        lambda x: len(str(x)) > excel_limit if isinstance(x, str) else False
                    )

                    if long_texts.any():
                        has_long_texts = True
                        summary_col = f"{col}_Summary"
                        excel_df[summary_col] = excel_df[col].apply(
                            lambda x: f"{str(x)[:excel_limit]}... (총 {len(str(x))} 자)"
                            if isinstance(x, str) and len(str(x)) > excel_limit else x
                        )

                        print(f"열 '{col}'에 {long_texts.sum()}개의 긴 텍스트(30,000자 초과)가 발견되어 '{summary_col}' 열에 요약 버전을 저장했습니다.")

            if not has_long_texts:
                print("모든 텍스트가 30,000자 미만으로, 원본 그대로 저장합니다.")

            excel_df.to_excel(excel_path, index=False)
            print(f"Excel 파일 저장 완료: {excel_path}")
            print(f"저장된 데이터 크기: {len(df)}행 x {len(df.columns)}열")

            return csv_path, excel_path
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {str(e)}")
            raise

class RationaleProcessor:

    @staticmethod
    def find_selection_columns(df: pd.DataFrame) -> List[str]:
        patterns = [
            r'.*Selected.*Rational[e]?.*Number.*',
            r'.*Selected.*Ration.*',
            r'.*Selection.*'
        ]

        for pattern in patterns:
            columns = [col for col in df.columns if re.match(pattern, col, re.IGNORECASE)]
            if columns:
                print(f"선택 컬럼 패턴 '{pattern}'에 맞는 {len(columns)}개 컬럼 발견:")
                for col in columns:
                    print(f"  - {col}")
                return columns

        raise ValueError("데이터프레임에서 선택된 Rationale 번호를 포함하는 컬럼을 찾을 수 없습니다.")

    @staticmethod
    def find_rationale_columns(df: pd.DataFrame) -> List[str]:
        patterns = [
            r'Rational_\d+$',
            r'Rationale_\d+$',
            r'Rationales_\d+$',
            r'Rational\d+$',
            r'Rationale\d+$'
        ]

        all_columns = []
        for pattern in patterns:
            columns = [col for col in df.columns if re.match(pattern, col, re.IGNORECASE)]
            if columns:
                print(f"Rationale 컬럼 패턴 '{pattern}'에 맞는 {len(columns)}개 컬럼 발견")
                all_columns.extend(columns)

        if all_columns:
            return sorted(all_columns)

        raise ValueError("데이터프레임에서 Rationale 내용이 포함된 컬럼을 찾을 수 없습니다.")

    @staticmethod
    def extract_number_from_column_name(column_name: str) -> Optional[int]:
        match = re.search(r'(\d+)$', column_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def get_selected_rationale_numbers(row: pd.Series, selection_columns: List[str]) -> List[int]:
        selected_numbers = []

        for column_name in selection_columns:
            if pd.notna(row[column_name]):
                try:
                    if isinstance(row[column_name], (int, float)):
                        number = int(row[column_name])
                        selected_numbers.append(number)
                    elif isinstance(row[column_name], str):
                        match = re.search(r'(\d+)', row[column_name])
                        if match:
                            number = int(match.group(1))
                            selected_numbers.append(number)
                except (ValueError, TypeError):
                    print(f"  경고: '{column_name}'의 값 '{row[column_name]}'을 숫자로 변환할 수 없습니다.")

        selected_numbers = sorted(set(selected_numbers))

        if not selected_numbers:
            participant_id = row.get('Participant_ID', 'Unknown')
            raise ValueError(f"참가자 ID {participant_id}의 선택 컬럼에서 유효한 Rationale 번호를 찾을 수 없습니다.")

        return selected_numbers

    @staticmethod
    def get_rationale_column_name(rationale_number: int, rationale_columns: List[str]) -> Optional[str]:
        patterns = [
            f'Rational_{rationale_number}$',
            f'Rationale_{rationale_number}$',
            f'Rationales_{rationale_number}$',
            f'Rational{rationale_number}$',
            f'Rationale{rationale_number}$'
        ]

        for pattern in patterns:
            for col in rationale_columns:
                if re.match(pattern, col, re.IGNORECASE):
                    return col

        return None

    @staticmethod
    def process_rationale_data(df: pd.DataFrame) -> pd.DataFrame:
        try:
            selection_columns = RationaleProcessor.find_selection_columns(df)
            rationale_columns = RationaleProcessor.find_rationale_columns(df)

            required_columns = ['Participant_ID', 'Transcript']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")

            optional_columns = ['Gender', 'PHQ_Binary', 'PHQ_Score']
            available_optional_cols = [col for col in optional_columns if col in df.columns]

            result_rows = []

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="선택된 Rationale 추출 중"):
                participant_id = row['Participant_ID']

                try:
                    selected_numbers = RationaleProcessor.get_selected_rationale_numbers(row, selection_columns)
                    print(f"참가자 ID {participant_id}: {len(selected_numbers)}개의 Rationale 선택됨: {selected_numbers}")

                    for rationale_number in selected_numbers:
                        column_name = RationaleProcessor.get_rationale_column_name(rationale_number, rationale_columns)

                        if not column_name:
                            print(f"  경고: 참가자 ID {participant_id}의 Rationale {rationale_number}에 해당하는 컬럼을 찾을 수 없습니다.")
                            continue

                        rationale_text = row[column_name]

                        if pd.isna(rationale_text):
                            print(f"  경고: 참가자 ID {participant_id}의 {column_name}이 비어 있습니다. 건너뜁니다.")
                            continue

                        new_row = {
                            'Participant_ID': participant_id,
                            'Selected_Rationale_Number': rationale_number,
                            'Selected_Rationale': rationale_text,
                            'Transcript': row['Transcript']
                        }

                        for col in available_optional_cols:
                            new_row[col] = row[col]

                        result_rows.append(new_row)

                except Exception as e:
                    print(f"참가자 ID {participant_id} 처리 중 오류 발생: {str(e)}")
                    continue

            if not result_rows:
                raise ValueError("선택된 Rationale를 찾을 수 없습니다.")

            result_df = pd.DataFrame(result_rows)

            column_order = ['Participant_ID', 'Gender', 'PHQ_Binary', 'PHQ_Score', 'Transcript', 'Selected_Rationale_Number', 'Selected_Rationale']
            available_columns = [col for col in column_order if col in result_df.columns]
            result_df = result_df[available_columns]

            print(f"선택된 Rationale 처리 완료: {len(result_df)}개 행 생성")

            return result_df

        except Exception as e:
            print(f"데이터 처리 중 오류 발생: {str(e)}")
            traceback.print_exc()
            raise

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='선택된 Rationale 추출 스크립트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        help='입력 파일 경로 (.xlsx, .xls, .csv 지원)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='출력 파일 경로 접두사 (확장자 없이, .csv와 .xlsx로 저장됨)'
    )
    parser.add_argument(
        '--dataset_folder',
        type=str,
        help='데이터셋 폴더 경로'
    )
    return parser.parse_args()

def main() -> None:
    print("선택된 Rationale 추출 스크립트 시작")

    try:
        args = parse_arguments()

        dataset_folder = args.dataset_folder or PathConfig.DEFAULT_DATASET_FOLDER
        input_path = args.input or PathConfig.get_full_path(PathConfig.DEFAULT_INPUT_FILE, dataset_folder)

        if args.output:
            output_base = args.output
        else:
            input_basename = os.path.basename(input_path)
            input_name = os.path.splitext(input_basename)[0]
            output_dir = os.path.join(dataset_folder, "results")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(output_dir, f"{PathConfig.DEFAULT_OUTPUT_PREFIX}_{input_name}_{timestamp}")

        csv_path, excel_path = PathConfig.get_output_paths(output_base)

        print(f"파일 로드 중: {input_path}")
        data_df = DatasetManager.load_data(input_path)

        print("선택된 Rationale 추출 중...")
        result_df = RationaleProcessor.process_rationale_data(data_df)

        if not result_df.empty:
            DatasetManager.save_data(result_df, csv_path, excel_path)
            print(f"모든 처리 완료.")
            print(f"CSV 결과 파일: {csv_path}")
            print(f"Excel 결과 파일: {excel_path}")
        else:
            print("선택된 Rationale이 없어 결과 파일을 저장하지 않았습니다.")

    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}")
        traceback.print_exc()

    print("스크립트 종료")

if __name__ == "__main__":
    main()
