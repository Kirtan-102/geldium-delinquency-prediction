"""
Data preprocessing and feature engineering for Geldium delinquency model.

Responsibilities:
- Load raw TSV/CSV data
- Clean column names and standardize categorical values
- Create interpretable features from payment history
- Impute / flag missing values
- Save processed datasets to data/processed/

Usage (from project root):
>>> from src.preprocess import load_and_preprocess
>>> df = load_and_preprocess("data/raw/Book1.txt", out_path="data/processed/processed.csv")
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_raw(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file {path} not found.")
    # Accept tab-separated or CSV depending on extension
    if path.suffix in (".txt", ".tsv"):
        df = pd.read_csv(path, sep="\t", engine="python")
    else:
        df = pd.read_csv(path)
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def normalize_employment_status(x) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    if s in ("emp", "employed", "employee"):
        return "employed"
    if "self" in s:
        return "self-employed"
    if "unemploy" in s:
        return "unemployed"
    if "retir" in s:
        return "retired"
    if "student" in s:
        return "student"
    return s

def payment_history_to_numeric(df: pd.DataFrame, prefix: str = "Month_") -> pd.DataFrame:
    """
    Map payment status (On-time/Late/Missed) to numeric risk indicators:
      On-time -> 0, Late -> 1, Missed -> 2
    Create summary features: count_missed, count_late, recent_missed, recent_late
    """
    df = df.copy()
    payment_cols = [c for c in df.columns if c.startswith(prefix)]
    mapping = {'On-time': 0, 'On time': 0, 'on-time': 0, 'On-time ': 0,
               'Late': 1, 'late': 1, 'Missed': 2, 'missed': 2}
    for c in payment_cols:
        df[c] = df[c].astype(str).str.strip().replace(mapping)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if payment_cols:
        df['count_missed'] = (df[payment_cols] == 2).sum(axis=1)
        df['count_late'] = (df[payment_cols] == 1).sum(axis=1)
        df['count_on_time'] = (df[payment_cols] == 0).sum(axis=1)
        # last 3 months recent behavior (if less than 3 cols, uses available)
        last3 = payment_cols[-3:]
        df['recent_missed'] = (df[last3] == 2).sum(axis=1)
        df['recent_late'] = (df[last3] == 1).sum(axis=1)
    else:
        df['count_missed'] = 0
        df['count_late'] = 0
        df['count_on_time'] = 0
        df['recent_missed'] = 0
        df['recent_late'] = 0
    return df

def coerce_numerics(df: pd.DataFrame, numeric_cols: Optional[list] = None) -> pd.DataFrame:
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = ['Age', 'Income', 'Credit_Score', 'Credit_Utilization', 'Missed_Payments',
                        'Loan_Balance', 'Debt_to_Income_Ratio', 'Account_Tenure']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def create_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['loan_balance_missing'] = df.get('Loan_Balance', pd.Series()).isna().astype(int) if 'Loan_Balance' in df else 1
    df['income_missing'] = df.get('Income', pd.Series()).isna().astype(int) if 'Income' in df else 1
    return df

def preprocess_core(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)
    # target - ensure presence
    if 'Delinquent_Account' not in df.columns:
        raise KeyError("Target column 'Delinquent_Account' missing from input data.")
    df['Delinquent_Account'] = pd.to_numeric(df['Delinquent_Account'], errors='coerce').fillna(0).astype(int)
    # employment normalization
    if 'Employment_Status' in df.columns:
        df['Employment_Status'] = df['Employment_Status'].apply(normalize_employment_status)
    else:
        df['Employment_Status'] = 'unknown'
    # credit card type & location standardization
    if 'Credit_Card_Type' in df.columns:
        df['Credit_Card_Type'] = df['Credit_Card_Type'].astype(str).str.strip().str.title()
    if 'Location' in df.columns:
        df['Location'] = df['Location'].astype(str).str.strip().str.title()
    # numerics
    df = coerce_numerics(df)
    # payment history to numeric indicators
    df = payment_history_to_numeric(df, prefix='Month_')
    # missing flags
    df = create_missing_flags(df)
    return df

def save_processed(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Processed data saved to {out}")

def load_and_preprocess(in_path: str, out_path: Optional[str] = None) -> pd.DataFrame:
    df = load_raw(in_path)
    df = preprocess_core(df)
    if out_path:
        save_processed(df, out_path)
    return df

if __name__ == "__main__":
    # Quick local run for debugging
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/Book1.txt")
    parser.add_argument("--output", default="data/processed/processed.csv")
    args = parser.parse_args()
    df = load_and_preprocess(args.input, args.output)
