import argparse
import os

from dotenv import load_dotenv
from openai import OpenAI


def load_system_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {file_path} 파일을 찾을 수 없습니다.")
        return ""
    except Exception as e:
        print(f"Error: 파일을 읽는 중 오류가 발생했습니다: {e}")
        return ""

USER_PROMPT = "Medical Rationale: "
MAX_TOKENS = 500
TEMPERATURE = 1.0
FEWSHOT_SUFFIX = (
    "\nFollowing examples 1 and 2, please create a similar rationale "
    "and diagnosis for example 3."
)


def main():
    parser = argparse.ArgumentParser(description="단일 프롬프트로 GPT-4o 호출")
    parser.add_argument("--prompt", "-p", default="system_prompt.txt",
                        help="시스템 프롬프트 텍스트 파일 경로")
    parser.add_argument("--model", "-m", default="gpt-4o", help="사용할 모델 이름")
    parser.add_argument("--few-shot", action="store_true",
                        help="실험 6(few-shot) 안내 문구를 프롬프트 끝에 덧붙입니다")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return 1

    system_prompt = load_system_prompt(args.prompt)
    if not system_prompt:
        print("시스템 프롬프트를 불러올 수 없어서 프로그램을 종료합니다.")
        return 1

    if args.few_shot:
        system_prompt += FEWSHOT_SUFFIX

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    exit(main())
