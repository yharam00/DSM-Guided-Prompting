import os
from dotenv import load_dotenv

from utils import LoggingManager, DatasetManager, ArgumentManager
from models import RationaleGenerator

def main() -> None:
    load_dotenv()

    logger = LoggingManager.setup_logging()
    logger.info("E-DAIC 트랜스크립트 기반 우울증 진단 의학적 근거 생성 프로그램 시작")

    try:
        args = ArgumentManager.parse_arguments()

        is_valid, error_message = ArgumentManager.validate_arguments(args, logger)
        if not is_valid:
            logger.error(error_message)
            return

        run_config = ArgumentManager.create_run_config(args)
        logger.info(f"실행 설정: {run_config}")

        logger.info(f"기존 데이터 로드 중... {run_config.input_path}")
        df = DatasetManager.load_existing_data(run_config.input_path, logger)

        logger.info(f"참가자 ID {run_config.start_id}부터 {run_config.end_id}까지 처리를 시작합니다...")
        updated_df = DatasetManager.process_participants_in_range(
            df,
            run_config,
            lambda row, model_name, rational_count, logger: RationaleGenerator.generate_rationales_for_participant(
                row, model_name, rational_count, logger, run_config.template_type
            ),
            logger
        )

        final_save_path = DatasetManager.save_progress(
            updated_df,
            run_config.output_path,
            f"ID{run_config.start_id}-{run_config.end_id}_final",
            logger
        )

        if final_save_path:
            logger.info(f"모든 요청된 참가자(ID {run_config.start_id}~{run_config.end_id}) 처리 완료!")
            logger.info(f"최종 결과 파일: {final_save_path}")
        else:
            logger.warning("참가자 처리는 완료되었으나 최종 저장에 실패했습니다.")

    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}", exc_info=True)

    logger.info("프로그램 종료")


if __name__ == "__main__":
    main()
