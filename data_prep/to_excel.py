import os
import argparse
from typing import List, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
import pandas as pd

DEFAULT_INPUT_FILE = 'data/daic_woz/gpt4o_exp1_test1.csv'
DEFAULT_OUTPUT_BASE = None

class OutputFormat(Enum):
    CSV = auto()
    EXCEL = auto()
    BOTH = auto()

    @classmethod
    def from_string(cls, value: str) -> 'OutputFormat':
        mapping = {'csv': cls.CSV, 'excel': cls.EXCEL, 'both': cls.BOTH}
        lower = value.lower()
        if lower not in mapping:
            raise ValueError(f"유효하지 않은 출력 형식입니다: {value}. 옵션: {', '.join(mapping.keys())}")
        return mapping[lower]


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        print(f"CSV 파일 로드 중: {file_path}")
        return pd.read_csv(file_path)
    if ext in ['.xlsx', '.xls']:
        print(f"Excel 파일 로드 중: {file_path}")
        return pd.read_excel(file_path)
    raise ValueError(f"지원하지 않는 파일 형식: {ext}")


def save_data(df: pd.DataFrame, base_path: str, fmt: OutputFormat) -> Tuple[Optional[str], Optional[str]]:
    csv_path = None
    excel_path = None
    dir_name = os.path.dirname(base_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    if fmt in (OutputFormat.CSV, OutputFormat.BOTH):
        csv_path = f"{base_path}.csv"
        df.to_csv(csv_path, index=False)
        print(f"CSV 저장: {csv_path}")
    if fmt in (OutputFormat.EXCEL, OutputFormat.BOTH):
        excel_path = f"{base_path}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"Excel 저장: {excel_path}")
    return csv_path, excel_path


def modify_phq_binary(df: pd.DataFrame, target_ids: List[int]) -> pd.DataFrame:
    if 'Participant_ID' not in df.columns or 'PHQ_Binary' not in df.columns:
        print(f"경고: 필요한 컬럼('Participant_ID', 'PHQ_Binary')이 없습니다. 데이터 수정 과정을 건너뜁니다.")
        return df
    result = df.copy()
    existing = set(result['Participant_ID'])
    to_mod = set(target_ids) & existing
    missing = set(target_ids) - existing
    if missing:
        print(f"경고: 데이터에 없는 ID: {', '.join(map(str, missing))}")
    count = 0
    for pid in to_mod:
        old = result.loc[result['Participant_ID']==pid, 'PHQ_Binary'].iloc[0]
        if old != 1:
            print(f"ID {pid}: {old} -> 1")
            count += 1
    result.loc[result['Participant_ID'].isin(to_mod), 'PHQ_Binary'] = 1
    print(f"총 {count}개 변경됨.")
    return result


def process_data(
    input_file: str,
    output_file: Optional[str],
    output_format: OutputFormat,
    target_ids: Optional[List[int]],
    no_timestamp: bool
) -> Tuple[Optional[str], Optional[str]]:
    if target_ids is None:
        target_ids = [325,335,352,356,380,386,409,413,418,422,459,483,633,682,691,696,709]
    df = load_data(input_file)
    print(f"Loaded {len(df)} rows x {len(df.columns)} cols.")
    mod_df = modify_phq_binary(df, target_ids)
    if output_file is None:
        base = os.path.splitext(input_file)[0]
        suffix = ""
        if not no_timestamp:
            suffix = "_modified"
            ts = datetime.now().strftime("_%Y%m%d_%H%M%S")
            suffix += ts
        output_base = f"{base}{suffix}"
    else:
        output_base = os.path.splitext(output_file)[0]
    return save_data(mod_df, output_base, output_format)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='데이터 처리 및 PHQ_Binary 수정',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, default=DEFAULT_INPUT_FILE,
                        help=f"입력 파일 경로 (default: {DEFAULT_INPUT_FILE})")
    parser.add_argument('--output', '-o', type=str, default=DEFAULT_OUTPUT_BASE,
                        help='출력 파일 베이스 경로 (default: None)')
    parser.add_argument('--format', '-f', type=str, choices=['csv','excel','both'], default='both',
                        help='출력 형식')
    parser.add_argument('--ids', type=int, nargs='+', help='수정할 Participant_ID 목록')
    parser.add_argument('--no-timestamp', action='store_true',
                        help='저장 파일명에 시간날짜를 붙이지 않습니다')
    return parser.parse_args()


def main():
    args = parse_arguments()
    ofmt = OutputFormat.from_string(args.format)
    try:
        csv_path, xlsx_path = process_data(
            input_file=args.input,
            output_file=args.output,
            output_format=ofmt,
            target_ids=args.ids,
            no_timestamp=args.no_timestamp
        )
        print("처리 완료!")
    except Exception as e:
        print(f"오류: {e}")
        return 1
    return 0

if __name__ == '__main__':
    exit(main())
