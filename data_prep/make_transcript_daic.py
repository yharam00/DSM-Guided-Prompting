import argparse
import sys
import re
from pathlib import Path
import pandas as pd


def normalize_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return str(int(m.group(1)))
        except Exception:
            pass
    try:
        return str(int(float(s)))
    except Exception:
        return s

def read_and_concat_text(csv_path, encoding="utf-8", sep=","):
    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
    cand_cols = ["Text", "text", "TEXT"]
    text_col = next((c for c in cand_cols if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"[{csv_path.name}]에 'Text' 열이 없습니다(가능 후보: {cand_cols}).")
    texts = (
        df[text_col]
        .dropna()
        .astype(str)
        .map(lambda s: re.sub(r"\s+", " ", s.strip()))
        .tolist()
    )
    return " ".join([t for t in texts if t])

def extract_id_from_filename(path: Path):
    m = re.match(r"^\s*(\d+)_", path.name)
    return m.group(1) if m else None

def find_pid_column(columns, user_specified=None):
    if user_specified:
        if user_specified in columns:
            return user_specified
        raise ValueError(f"[에러] 지정한 --pid-col '{user_specified}'을(를) 찾을 수 없습니다. 실제 컬럼: {list(columns)}")
    candidates = [
        "Participant_ID", "participant_id", "PARTICIPANT_ID",
        "ParticipantId", "participantId",
        "Pariticipant_ID",
        "ID", "id"
    ]
    for c in candidates:
        if c in columns:
            return c
    for c in columns:
        lc = c.lower()
        if ("participant" in lc or "pariticipant" in lc) and "id" in lc:
            return c
    raise ValueError(f"[에러] Participant_ID 컬럼을 찾지 못했습니다. 후보명 중 하나로 맞춰주세요: {candidates} 또는 --pid-col로 지정하세요.")


def main():
    ap = argparse.ArgumentParser(description="Transcript CSV들을 명단 CSV에 병합")
    ap.add_argument("--roster", required=True, help="명단 CSV 파일 경로 (Participant_ID 열 존재 필요)")
    ap.add_argument("--folder", required=True, help="Transcript CSV들이 있는 폴더 경로")
    ap.add_argument("--pattern", default="*_Transcript.csv", help="Transcript 파일 패턴 (기본: *_Transcript.csv)")
    ap.add_argument("--encoding", default="utf-8", help="파일 인코딩 (기본: utf-8)")
    ap.add_argument("--sep", default=",", help="CSV 구분자 (기본: ,)")
    ap.add_argument("--inplace", action="store_true", help="원본 명단 CSV를 덮어쓰기")
    ap.add_argument("--output", help="결과 저장 경로(미지정 시 원본명 + _with_transcripts.csv)")
    ap.add_argument("--merge-mode", choices=["overwrite", "fillna"], default="overwrite",
                    help="Transcript 병합 방식: overwrite(덮어씀) 또는 fillna(기존 비어있는 곳만 채움)")
    ap.add_argument("--pid-col", help="명단 CSV의 Participant_ID 컬럼명(자동 인식 실패 시 지정)")
    args = ap.parse_args()

    folder = Path(args.folder)
    roster_path = Path(args.roster)

    if not roster_path.exists():
        print(f"[에러] 명단 파일을 찾을 수 없습니다: {roster_path}", file=sys.stderr)
        sys.exit(1)
    if not folder.is_dir():
        print(f"[에러] 폴더 경로가 올바르지 않습니다: {folder}", file=sys.stderr)
        sys.exit(1)

    roster = pd.read_csv(roster_path, encoding=args.encoding, sep=args.sep)

    pid_col = find_pid_column(roster.columns, user_specified=args.pid_col)

    roster["_PID_norm"] = roster[pid_col].map(normalize_id)

    files = sorted(folder.glob(args.pattern))
    if not files:
        print(f"[경고] 패턴에 맞는 파일이 없습니다: {args.pattern} (폴더: {folder})", file=sys.stderr)

    id_to_texts = {}
    for f in files:
        pid = extract_id_from_filename(f)
        if not pid:
            print(f"[건너뜀] 숫자ID_ 로 시작하지 않는 파일: {f.name}", file=sys.stderr)
            continue
        try:
            long_text = read_and_concat_text(f, encoding=args.encoding, sep=args.sep)
        except Exception as e:
            print(f"[에러] {f.name} 처리 실패: {e}", file=sys.stderr)
            continue
        if long_text.strip():
            id_to_texts.setdefault(normalize_id(pid), []).append(long_text)

    id_to_merged = {pid: " ".join(txts) for pid, txts in id_to_texts.items()}

    if "Transcript" not in roster.columns:
        roster["Transcript"] = pd.NA

    applied = 0
    for pid_norm, merged_text in id_to_merged.items():
        mask = roster["_PID_norm"] == pid_norm
        if not mask.any():
            print(f"[건너뜀] 명단에서 Participant_ID='{pid_norm}'(정규화) 를 찾지 못했습니다.", file=sys.stderr)
            continue
        if args.merge_mode == "fillna":
            to_fill = roster.loc[mask, "Transcript"].isna()
            roster.loc[mask & to_fill, "Transcript"] = merged_text
            applied += int((mask & to_fill).sum())
        else:
            roster.loc[mask, "Transcript"] = merged_text
            applied += int(mask.sum())

    roster.drop(columns=["_PID_norm"], inplace=True)

    if args.inplace and args.output:
        print("[경고] --inplace와 --output을 동시에 지정하면 --inplace가 우선입니다.", file=sys.stderr)

    if args.inplace:
        out_path = roster_path
    else:
        out_path = Path(args.output) if args.output else roster_path.with_name(
            roster_path.stem + "_with_transcripts" + roster_path.suffix
        )

    roster.to_csv(out_path, index=False, encoding=args.encoding, sep=args.sep)

    print(f"[완료] {applied}개 행에 Transcript를 적용했습니다.")
    print(f"[저장] {out_path}")

if __name__ == "__main__":
    main()
