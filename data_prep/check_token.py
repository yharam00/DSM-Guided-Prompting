import os
import pandas as pd
import logging
import argparse
import tiktoken
from datetime import datetime
from typing import Optional

def setup_logging() -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"check_token_{timestamp}.log")

    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def calculate_token_length(text: str, model_name: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)

def process_excel_file(input_path: str, output_path: str, logger: logging.Logger) -> None:
    try:
        logger.info(f"엑셀 파일 로드 중: {input_path}")
        df = pd.read_excel(input_path)

        logger.info(f"로드된 데이터 크기: {len(df)}행 x {len(df.columns)}열")

        if 'Prompt' not in df.columns:
            logger.error("'Prompt' 열이 입력 파일에 존재하지 않습니다.")
            return

        logger.info("각 프롬프트의 토큰 수 계산 중...")

        token_counts = []

        for idx, row in df.iterrows():
            if pd.notna(row['Prompt']):
                token_count = calculate_token_length(row['Prompt'])
                token_counts.append(token_count)
                logger.info(f"행 {idx+1}: 토큰 수 = {token_count}")
            else:
                token_counts.append(0)
                logger.warning(f"행 {idx+1}: 프롬프트가 없습니다.")

        df['Calculated_Token_Count'] = token_counts

        if 'Token_Length' in df.columns:
            df['Token_Difference'] = df['Calculated_Token_Count'] - df['Token_Length']

            mean_diff = df['Token_Difference'].mean()
            max_diff = df['Token_Difference'].max()
            min_diff = df['Token_Difference'].min()

            logger.info(f"토큰 수 차이 통계:")
            logger.info(f"  평균 차이: {mean_diff:.2f}")
            logger.info(f"  최대 차이: {max_diff}")
            logger.info(f"  최소 차이: {min_diff}")

        logger.info(f"결과를 저장 중: {output_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(output_path, index=False)

        avg_token = df['Calculated_Token_Count'].mean()
        max_token = df['Calculated_Token_Count'].max()
        min_token = df['Calculated_Token_Count'].min()

        logger.info("처리 완료:")
        logger.info(f"총 프롬프트 수: {len(df)}")
        logger.info(f"평균 토큰 길이: {avg_token:.2f}")
        logger.info(f"최대 토큰 길이: {max_token}")
        logger.info(f"최소 토큰 길이: {min_token}")

    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생: {str(e)}", exc_info=True)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='엑셀 파일의 Prompt 열에 대한 토큰 수 확인 스크립트',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 엑셀 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='출력 엑셀 파일 경로 (기본값: input 파일명에 _checked 추가)'
    )
    return parser.parse_args()

def main() -> None:
    logger = setup_logging()
    logger.info("토큰 수 확인 스크립트 시작")

    try:
        args = parse_arguments()

        input_path = args.input

        if args.output:
            output_path = args.output
        else:
            base_name, ext = os.path.splitext(input_path)
            output_path = f"{base_name}_checked{ext}"

        process_excel_file(input_path, output_path, logger)

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}", exc_info=True)

    logger.info("스크립트 종료")

if __name__ == "__main__":
    main()
