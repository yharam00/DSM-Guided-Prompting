import argparse
from pathlib import Path
import sys
import pandas as pd

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}
CSV_EXTS = {".csv", ".tsv", ".txt"}

def parse_args():
    p = argparse.ArgumentParser(description="Keep only the first row per Participant_ID; save using same format as input.")
    p.add_argument("-i", "--input", required=True, help="Path to input table (Excel .xlsx/.xls/.xlsm or CSV/TSV .csv/.tsv/.txt).")
    p.add_argument("-s", "--sheet", default=None, help="(Excel input only) Sheet name or index (default: first sheet).")
    p.add_argument("--id-col", default="Participant_ID", help="Column name to deduplicate on (default: Participant_ID).")
    p.add_argument("--dtype", choices=["int", "string"], default="int",
                   help="Interpret ID column as 'int' (default) or 'string' (preserve leading zeros, trim spaces).")
    p.add_argument("--in-sep", default=None, help="(CSV input only) Input delimiter. Default: ',' for .csv, '\t' for .tsv, simple sniff otherwise.")
    p.add_argument("--out-sep", default=None, help="(CSV output only) Output delimiter. Default: same as input delimiter.")
    p.add_argument("-o", "--out", default=None, help="Output path (default: <input_basename>.dedup<same_ext>).")
    return p.parse_args()

def ensure_col_exists(df: pd.DataFrame, col: str, source_label: str):
    if col not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise KeyError(f"Column '{col}' not found in {source_label}. Available columns: {cols}")

def sniff_sep(path: Path, preferred: str | None):
    if preferred is not None:
        return preferred
    ext = path.suffix.lower()
    if ext == ".tsv":
        return "\t"
    if ext == ".csv":
        return ","
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline()
        if head.count("\t") > head.count(","):
            return "\t"
    except Exception:
        pass
    return ","

def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    ext = in_path.suffix.lower()
    is_excel = ext in EXCEL_EXTS
    is_csvlike = ext in CSV_EXTS
    if not (is_excel or is_csvlike):
        print(f"ERROR: Unsupported input extension '{ext}'. Supported: {sorted(EXCEL_EXTS|CSV_EXTS)}", file=sys.stderr)
        sys.exit(1)

    id_col = args.id_col

    try:
        if is_excel:
            dtype_map = {id_col: "string"} if args.dtype == "string" else {id_col: "Int64"}
            if args.sheet is None:
                sheet_arg = 0
            else:
                try:
                    sheet_arg = int(args.sheet)
                except ValueError:
                    sheet_arg = args.sheet
            df = pd.read_excel(in_path, sheet_name=sheet_arg, dtype=dtype_map, engine=None)
        else:
            in_sep = sniff_sep(in_path, args.in_sep)
            dtype_map = {id_col: "string"} if args.dtype == "string" else None
            df = pd.read_csv(in_path, sep=in_sep, dtype=dtype_map, engine="python")
    except Exception as e:
        print(f"ERROR: Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ensure_col_exists(df, id_col, f"input '{in_path.name}'")
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dtype == "int":
        key = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    else:
        key = df[id_col].astype("string").str.strip()

    before = len(df)
    df = df.assign(__dedupe_key__=key)
    deduped = df.drop_duplicates(subset="__dedupe_key__", keep="first", ignore_index=False).drop(columns="__dedupe_key__")
    after = len(deduped)

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + in_path.suffix)

    try:
        if is_excel:
            deduped.to_excel(out_path, index=False)
        else:
            in_sep = sniff_sep(in_path, args.in_sep)
            out_sep = args.out_sep if args.out_sep is not None else in_sep
            deduped.to_csv(out_path, index=False, sep=out_sep, encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    removed = before - after
    print(f"Done. Kept first occurrence per '{id_col}'. Removed {removed} duplicates. Wrote {after} rows to: {out_path}")

if __name__ == "__main__":
    main()
