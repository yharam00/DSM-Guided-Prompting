import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from string import Template
from openai import OpenAI
import pandas as pd
import argparse

from config import ModelConfig, ModelsConfig, DefaultConfig, PromptTemplateConfig, RunConfig
from utils import ArgumentManager


class PromptGenerator:

    @staticmethod
    def create_prompt(
        transcript: str,
        gender: Optional[str] = None,
        phq_binary: Optional[int] = None,
        include_dsm: bool = False,
        template_type: str = "main"
    ) -> str:
        if template_type == "template_for_experiment_2":
            template_str = PromptTemplateConfig.template_for_experiment_2
        elif template_type == "gender_dsm":
            template_str = PromptTemplateConfig.GENDER_DSM_TEMPLATE
        elif template_type == "transcript_depression":
            template_str = PromptTemplateConfig.TRANSCRIPT_DEPRESSION_TEMPLATE
        elif template_type == "template_for_experiment_4":
            template_str = PromptTemplateConfig.template_for_experiment_4
        elif template_type == "template_for_experiment_4_with_ptsd":
            template_str = PromptTemplateConfig.template_for_experiment_4_with_ptsd
        else:
            template_str = PromptTemplateConfig.MAIN_TEMPLATE

        template_str = template_str.replace("{{", "${").replace("}}", "}")
        template = Template(template_str)

        substitution_values = {"transcript": transcript}

        if gender:
            substitution_values["gender"] = gender

        if phq_binary is not None:
            depression_status = "depressed" if phq_binary == 1 else "non-depressed"
            substitution_values["depression_status"] = depression_status

        if include_dsm or template_type in ["main", "gender_dsm", "template_for_experiment_4", "template_for_experiment_4_with_ptsd"]:
            substitution_values["dsm5_criteria"] = PromptTemplateConfig.DSM5_CRITERIA

        if template_type == "template_for_experiment_4_with_ptsd":
            substitution_values["dsm5_ptsd"] = PromptTemplateConfig.DSM5_PTSD

        try:
            prompt = template.substitute(**substitution_values)
            return prompt
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"템플릿 '{template_type}'에 필요한 값 '{missing_key}'이(가) 제공되지 않았습니다.")


class ModelClient:

    @staticmethod
    def get_api_client(model_name: str) -> Tuple[Optional[OpenAI], Optional[str]]:
        model_config = ModelsConfig.get_model_config(model_name)
        if not model_config:
            return None, f"지원되지 않는 모델: {model_name}"

        api_key = os.getenv(model_config.api_key_env)

        if not api_key:
            return None, f"{model_config.api_key_env} 환경 변수가 설정되지 않았습니다"

        try:
            if model_config.base_url:
                client = OpenAI(api_key=api_key, base_url=model_config.base_url)
            else:
                client = OpenAI(api_key=api_key)

            return client, None
        except Exception as e:
            return None, f"API 클라이언트 생성 중 오류: {str(e)}"

    @staticmethod
    def get_model_messages(prompt: str, model_name: str) -> List[Dict[str, str]]:
        if model_name == "deepseek-reasoner":
            return [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Medical Rationale: "}
            ]
        else:
            return [
                {"role": "developer", "content": prompt},
                {"role": "user", "content": "Medical Rationale: "}
            ]


class RationaleGenerator:

    @staticmethod
    def get_medical_rationale(prompt: str, model_name: str) -> Tuple[Optional[str], Optional[str]]:
        client, error = ModelClient.get_api_client(model_name)
        if error:
            return None, error

        try:
            messages = ModelClient.get_model_messages(prompt, model_name)

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=DefaultConfig.MAX_TOKENS,
                temperature=DefaultConfig.TEMPERATURE
            )
            return response.choices[0].message.content, None
        except Exception as e:
            return None, f"모델 ({model_name}) API 호출 중 오류 발생: {str(e)}"

    @staticmethod
    def generate_rationales_for_participant(
        row: Any,
        model_name: str,
        rational_count: int = DefaultConfig.RATIONAL_COUNT,
        logger: Optional[logging.Logger] = None,
        template_type: str = "main"
    ) -> Dict[str, Optional[str]]:
        participant_id = row['Participant_ID']
        rationale_dict = {}

        if template_type == "template_for_experiment_2":
            prompt = PromptGenerator.create_prompt(
                transcript=row['Transcript'],
                template_type=template_type
            )
        elif template_type == "gender_dsm":
            prompt = PromptGenerator.create_prompt(
                transcript=row['Transcript'],
                gender=row['Gender'],
                include_dsm=True,
                template_type=template_type
            )
        elif template_type == "transcript_depression":
            prompt = PromptGenerator.create_prompt(
                transcript=row['Transcript'],
                phq_binary=row['PHQ_Binary'],
                gender=row['Gender'],
                include_dsm=True,
                template_type=template_type
            )
        elif template_type == "template_for_experiment_4":
            prompt = PromptGenerator.create_prompt(
                transcript=row['Transcript'],
                gender=row['Gender'],
                include_dsm=True,
                template_type=template_type
            )
        elif template_type == "template_for_experiment_4_with_ptsd":
            prompt = PromptGenerator.create_prompt(
                transcript=row['Transcript'],
                gender=row['Gender'],
                include_dsm=True,
                template_type=template_type
            )
        else:
            prompt = PromptGenerator.create_prompt(
                transcript=row['Transcript'],
                gender=row['Gender'],
                phq_binary=row['PHQ_Binary'],
                include_dsm=True,
                template_type="main"
            )

        rationale_dict["Prompt"] = prompt

        if logger:
            logger.info(f"템플릿 유형: {template_type}")
            logger.info(f"생성된 프롬프트 시작 부분: {prompt[:100]}...")

        for rational_num in range(1, rational_count + 1):
            column_name = f'Rational_{rational_num}'

            if column_name in row and pd.notna(row[column_name]):
                if logger:
                    logger.info(f"참가자 {participant_id}의 {column_name}이 이미 존재합니다. 건너뜁니다.")
                rationale_dict[column_name] = row[column_name]
                continue

            try:
                rationale, error = RationaleGenerator.get_medical_rationale(prompt, model_name)

                if error:
                    if logger:
                        logger.error(f"참가자 {participant_id}의 {column_name} 생성 중 오류: {error}")
                    rationale_dict[column_name] = None
                else:
                    rationale_dict[column_name] = rationale
                    if logger:
                        logger.info(f"참가자 {participant_id}의 {column_name} 생성 완료")

                time.sleep(DefaultConfig.SLEEP_TIME)

            except Exception as e:
                if logger:
                    logger.error(f"참가자 {participant_id}의 {column_name} 생성 중 오류: {str(e)}")
                rationale_dict[column_name] = None

        return rationale_dict

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description='E-DAIC 데이터셋 특정 참가자 범위에 대한 의학적 근거 생성기',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
