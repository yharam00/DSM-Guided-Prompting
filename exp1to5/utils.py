import os
import logging
import pandas as pd
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm

from config import PathConfig, RunConfig, ModelsConfig


class LoggingManager:

    @staticmethod
    def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"rational_generation_{timestamp}.log"

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


class DatasetManager:

    @staticmethod
    def load_existing_data(csv_path: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path)
            if logger:
                logger.info(f"기존 데이터 로드 완료: {len(df)}개 행, {len(df.columns)}개 열")

                rational_columns = [col for col in df.columns if col.startswith('Rational_')]
                logger.info(f"이미 생성된 Rational 열: {len(rational_columns)}개")

                min_id = df['Participant_ID'].min()
                max_id = df['Participant_ID'].max()
                logger.info(f"참가자 ID 범위: {min_id}부터 {max_id}까지, 총 {len(df)}명")

            return df
        except FileNotFoundError:
            error_msg = f"CSV 파일을 찾을 수 없습니다: {csv_path}"
            if logger:
                logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except pd.errors.EmptyDataError:
            error_msg = f"CSV 파일이 비어있습니다: {csv_path}"
            if logger:
                logger.error(error_msg)
            raise pd.errors.EmptyDataError(error_msg)

    @staticmethod
    def save_progress(
        df: pd.DataFrame,
        output_path: str,
        checkpoint_name: str,
        logger: Optional[logging.Logger] = None
    ) -> Optional[str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = os.path.splitext(output_path)[0]
        file_ext = os.path.splitext(output_path)[1]
        new_file_path = f"{file_name}_{checkpoint_name}_{timestamp}{file_ext}"

        try:
            df.to_csv(new_file_path, index=False)
            df.to_csv(output_path, index=False)

            if logger:
                logger.info(f"진행 상황 저장 완료: {new_file_path} (백업) 및 {output_path} (원본)")
            return new_file_path
        except Exception as e:
            if logger:
                logger.error(f"진행 상황 저장 중 오류 발생: {str(e)}")
            return None

    @staticmethod
    def process_participants_in_range(
        df: pd.DataFrame,
        config: RunConfig,
        generate_rationales_func: Any,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        selected_participants = df[
            (df['Participant_ID'] >= config.start_id) &
            (df['Participant_ID'] <= config.end_id)
        ]

        if selected_participants.empty:
            if logger:
                logger.warning(f"ID {config.start_id}부터 {config.end_id}까지의 참가자가 없습니다.")
            return df

        if logger:
            logger.info(f"선택된 참가자 수: {len(selected_participants)}명")
            logger.info(f"사용 모델: {config.model_name}")

        result_df = df.copy()

        for idx, row in tqdm(selected_participants.iterrows(), total=len(selected_participants),
                            desc=f"참가자 ID {config.start_id}~{config.end_id} 처리 중"):
            participant_id = row['Participant_ID']
            if logger:
                logger.info(f"=== 참가자 {participant_id} 처리 시작 ===")

            rationale_dict = generate_rationales_func(
                row,
                config.model_name,
                config.rational_count,
                logger
            )

            for column, value in rationale_dict.items():
                if value is not None:
                    result_df.loc[idx, column] = value

            DatasetManager.save_progress(
                result_df,
                config.output_path,
                f"ID{participant_id}",
                logger
            )

            if logger:
                logger.info(f"=== 참가자 {participant_id} 처리 완료 ===")

        return result_df


class ArgumentManager:

    @staticmethod
    def validate_arguments(args: argparse.Namespace, logger: Optional[logging.Logger] = None) -> Tuple[bool, str]:
        if args.start_id < 0:
            return False, "오류: 시작 참가자 ID는 0 이상이어야 합니다."

        if args.end_id < args.start_id:
            return False, f"오류: 종료 참가자 ID는 시작 ID({args.start_id}) 이상이어야 합니다."

        if args.input and not os.path.exists(args.input):
            return False, f"오류: 입력 파일이 존재하지 않습니다: {args.input}"

        supported_models = ModelsConfig.get_supported_models()
        if args.model not in supported_models:
            return False, f"오류: 지원되지 않는 모델입니다: {args.model}. 지원 모델: {', '.join(supported_models)}"

        model_config = ModelsConfig.get_model_config(args.model)
        if not model_config:
            return False, f"오류: 모델 설정을 찾을 수 없습니다: {args.model}"

        if not os.getenv(model_config.api_key_env):
            return False, f"오류: {model_config.api_key_env} 환경 변수가 설정되지 않았습니다."

        return True, ""

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description='E-DAIC 데이터셋 특정 참가자 범위에 대한 의학적 근거 생성기',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            '--start_id',
            type=int,
            default=302,
            help='시작 참가자 ID'
        )
        parser.add_argument(
            '--end_id',
            type=int,
            default=718,
            help='종료 참가자 ID'
        )
        parser.add_argument(
            '--input',
            type=str,
            help='입력 CSV 파일 경로'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='출력 CSV 파일 경로'
        )
        parser.add_argument(
            '--model',
            type=str,
            default="gpt-4o",
            choices=ModelsConfig.get_supported_models(),
            help='사용할 AI 모델'
        )
        parser.add_argument(
            '--count',
            type=int,
            default=30,
            help='생성할 Rational의 수'
        )
        parser.add_argument(
            '--template',
            type=str,
            default="main",
            choices=["main", "template_for_experiment_2", "gender_dsm", "transcript_depression", "template_for_experiment_4", "template_for_experiment_4_with_ptsd"],
            help='사용할 프롬프트 템플릿 유형'
        )
        return parser.parse_args()

    @staticmethod
    def get_file_paths(args: argparse.Namespace) -> Tuple[str, str]:
        env_dataset_folder = os.getenv("DATASET_FOLDER")
        input_path = args.input or PathConfig.get_default_input_path(env_dataset_folder)

        if not args.output:
            output_path = PathConfig.get_default_output_path(env_dataset_folder, args.template)
        else:
            output_path = args.output

        return input_path, output_path

    @staticmethod
    def create_run_config(args: argparse.Namespace) -> RunConfig:
        input_path, output_path = ArgumentManager.get_file_paths(args)

        return RunConfig(
            start_id=args.start_id,
            end_id=args.end_id,
            input_path=input_path,
            output_path=output_path,
            model_name=args.model,
            rational_count=args.count,
            template_type=args.template
        )
