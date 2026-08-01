import os
import glob
import pandas as pd
import argparse
from tqdm import tqdm


def read_metadata(file_path):
    return pd.read_csv(file_path)


def process_transcript_file(file_path):
    participant_id = os.path.basename(file_path).split('_')[0]

    try:
        transcript_df = pd.read_csv(file_path)

        if 'Text' in transcript_df.columns:
            combined_text = ' '.join(transcript_df['Text'].dropna().astype(str))
            return participant_id, combined_text
        else:
            print(f"경고: {file_path}에 Text 컬럼이 없습니다.")
            return participant_id, ""
    except Exception as e:
        print(f"오류: {file_path} 처리 중 예외 발생 - {e}")
        return participant_id, ""


def process_dataset(dataset_type='test'):
    base_dir = "data"
    metadata_file = os.path.join(base_dir, f"e_daic_metadata/{dataset_type}_split.csv")
    transcript_dir = os.path.join(base_dir, "e_daic_text_only")
    output_file = os.path.join(base_dir, f"e_daic_transcript_{dataset_type}.csv")

    print(f"{dataset_type} 메타데이터 읽는 중...")
    metadata_df = read_metadata(metadata_file)

    transcript_files = glob.glob(os.path.join(transcript_dir, "*_Transcript.csv"))

    print(f"{len(transcript_files)}개의 트랜스크립트 파일 처리 중...")
    transcript_data = []

    for file_path in tqdm(transcript_files):
        participant_id, combined_text = process_transcript_file(file_path)
        transcript_data.append({
            'Participant_ID': participant_id,
            'Transcript': combined_text
        })

    transcript_df = pd.DataFrame(transcript_data)

    metadata_df['Participant_ID'] = metadata_df['Participant_ID'].astype(str)
    transcript_df['Participant_ID'] = transcript_df['Participant_ID'].astype(str)

    print("메타데이터와 트랜스크립트 병합 중...")
    merged_df = pd.merge(
        metadata_df,
        transcript_df,
        on='Participant_ID',
        how='inner'
    )

    print(f"결과를 {output_file}에 저장 중...")
    merged_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"처리 완료! 총 {len(merged_df)}개의 {dataset_type} 레코드가 생성됨.")


def main():
    parser = argparse.ArgumentParser(description='e_daic 데이터셋의 트랜스크립트와 메타데이터를 결합합니다.')
    parser.add_argument('--type', type=str, choices=['test', 'train', 'both'], default='both',
                      help='처리할 데이터셋 유형 (test, train, 또는 both)')

    args = parser.parse_args()

    if args.type == 'test' or args.type == 'both':
        process_dataset('test')

    if args.type == 'train' or args.type == 'both':
        process_dataset('train')


if __name__ == "__main__":
    main()
