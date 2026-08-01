import argparse
from pathlib import Path
import sys
import pandas as pd

EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}
CSV_EXTS = {".csv", ".tsv", ".txt"}

def parse_args():
    p = argparse.ArgumentParser(description="Keep only rows from a table whose Participant_ID appears in an IDs CSV.")
    p.add_argument("-i", "--input", required=True, help="Path to input table (Excel .xlsx/.xls/.xlsm or CSV/TSV .csv/.tsv/.txt).")
    p.add_argument("--ids", required=True, help="Path to CSV file containing the ID list (must have the ID column).")
    p.add_argument("-s", "--sheet", default=None, help="(Excel input only) Sheet name or index (default: first sheet).")
    p.add_argument("--id-col", default="Participant_ID", help="Column name to match on (default: Participant_ID).")
    p.add_argument("--ids-sep", default=",", help="Delimiter for the IDs CSV (default: ',').")
    p.add_argument("--in-sep", default=None, help="(CSV input only) Input delimiter. Default: ',' for .csv, '\t' for .tsv, auto sniff otherwise.")
    p.add_argument("--out-sep", default=None, help="(CSV output only) Output delimiter. Default: same as input.")
    p.add_argument("--dtype", choices=["int", "string"], default="int",
                   help="How to read the ID column: 'int' (default) for numeric IDs, 'string' for IDs with leading zeros.")
    p.add_argument("--no-sort", action="store_true", help="Do not sort the output by the ID column.")
    p.add_argument("-o", "--out", default=None, help="Output path (default: <input_basename>.filtered<same_ext>).")
    return p.parse_args()

def ensure_col_exists(df: pd.DataFrame, col: str, source_label: str):
    if col not in df.columns:
        cols = ", ".join(map(str, df.columns))
        raise KeyError(f"Column '{col}' not found in {source_label}. Available columns: {cols}")

def read_ids(ids_path: str, id_col: str, sep: str, dtype_mode: str) -> pd.Series:
    try:
        if dtype_mode == "int":
            ids = pd.read_csv(ids_path, sep=sep, usecols=[id_col], dtype={id_col: "Int64"})
            ids = ids[id_col].dropna().astype("Int64")
        else:
            ids = pd.read_csv(ids_path, sep=sep, usecols=[id_col], dtype={id_col: "string"})
            ids = ids[id_col].astype("string").str.strip().dropna()
    except ValueError:
        df_all = pd.read_csv(ids_path, sep=sep, dtype="string" if dtype_mode == "string" else None)
        ensure_col_exists(df_all, id_col, f"IDs CSV '{ids_path}'")
        ids = df_all[id_col]
        if dtype_mode == "int":
            ids = pd.to_numeric(ids, errors="coerce").astype("Int64").dropna()
        else:
            ids = ids.astype("string").str.strip().dropna()
    return ids

def sniff_sep(path: Path, default: str):
    if default is not None:
        return default
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
    ids_path = Path(args.ids)
    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    if not ids_path.exists():
        print(f"ERROR: IDs CSV not found: {ids_path}", file=sys.stderr)
        sys.exit(1)

    id_col = args.id_col

    ids_series = read_ids(str(ids_path), id_col=id_col, sep=args.ids_sep, dtype_mode=args.dtype)
    ids_set = set(ids_series.dropna().tolist())
    if len(ids_set) == 0:
        print(f"WARNING: No valid IDs found in IDs CSV column '{id_col}'. Output may be empty.", file=sys.stderr)

    ext = in_path.suffix.lower()
    is_excel = ext in EXCEL_EXTS
    is_csvlike = ext in CSV_EXTS

    if not (is_excel or is_csvlike):
        print(f"ERROR: Unsupported input extension '{ext}'. Supported: {sorted(EXCEL_EXTS|CSV_EXTS)}", file=sys.stderr)
        sys.exit(1)

    try:
        if is_excel:
            excel_dtype = {id_col: "string"} if args.dtype == "string" else {id_col: "Int64"}
            if args.sheet is None:
                sheet_arg = 0
            else:
                try:
                    sheet_arg = int(args.sheet)
                except ValueError:
                    sheet_arg = args.sheet
            df_in = pd.read_excel(in_path, sheet_name=sheet_arg, dtype=excel_dtype, engine=None)
        else:
            in_sep = sniff_sep(in_path, args.in_sep)
            dtype_map = {id_col: "string"} if args.dtype == "string" else None
            df_in = pd.read_csv(in_path, sep=in_sep, dtype=dtype_map, engine="python")
    except Exception as e:
        print(f"ERROR: Failed to read input: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ensure_col_exists(df_in, id_col, f"input '{in_path.name}'")
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dtype == "int":
        series = pd.to_numeric(df_in[id_col], errors="coerce").astype("Int64")
        mask = series.isin(ids_set)
    else:
        series = df_in[id_col].astype("string").str.strip()
        ids_set = set(pd.Series(list(ids_set), dtype="string").astype("string").str.strip().tolist())
        mask = series.isin(ids_set)

    filtered = df_in.loc[mask].copy()

    if not args.no_sort and id_col in filtered.columns:
        try:
            if args.dtype == "int":
                filtered = filtered.sort_values(by=id_col, kind="stable")
            else:
                filtered = filtered.sort_values(by=id_col, kind="stable", key=lambda s: s.astype("string").str.zfill(16))
        except Exception:
            pass

    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_DAIC_test" + in_path.suffix)

    try:
        if is_excel:
            filtered.to_excel(out_path, index=False)
        else:
            in_sep = sniff_sep(in_path, args.in_sep)
            out_sep = args.out_seP if args.out_sep is not None else in_sep
            filtered.to_csv(out_path, index=False, sep=out_sep, encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(filtered)} rows to: {out_path}")

if __name__ == "__main__":
    main()
