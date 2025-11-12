# prepare_yoochoose.py
import pandas as pd
from pathlib import Path
from datetime import timedelta
import random

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def load_yoochoose_clicks(path: str, sample_size: int = 200_000, seed: int = 42):
    print(" Loading YooChoose click dataset ...")
    cols = ['session_id', 'timestamp', 'item_id', 'category']
    df = pd.read_csv(path, header=None, names=cols,nrows=100_000, parse_dates=[1], low_memory=False)
    print(f" Loaded {len(df):,} events, {df['session_id'].nunique():,} sessions, {df['item_id'].nunique():,} items")
    return df

def filter_and_build_sessions(df, min_session_len=2):
    print(f"\n Filtering sessions (min_len={min_session_len}) ...")
    session_lens = df.groupby('session_id').size()
    good_sessions = session_lens[session_lens >= min_session_len].index

    before = df['session_id'].nunique()
    df = df[df['session_id'].isin(good_sessions)]
    after = df['session_id'].nunique()
    print(f"• removed short sessions: {before - after:,} | kept: {after:,}")
    print(f" Final dataset: {len(df):,} events")
    return df

def make_train_test(df, test_last_days=1):
    print(f"\n Train/Test split: last {test_last_days} day(s) for test ...")
    max_ts = df['timestamp'].max()
    cutoff = max_ts - timedelta(days=test_last_days)

    last_session = df.groupby('session_id')['timestamp'].max()
    test_sessions = last_session[last_session >= cutoff].index
    train_sessions = last_session[last_session < cutoff].index

    train = df[df['session_id'].isin(train_sessions)]
    test = df[df['session_id'].isin(test_sessions)]

    print(f" Train: {train['session_id'].nunique():,} sessions | {len(train):,} events")
    print(f" Test:  {test['session_id'].nunique():,} sessions | {len(test):,} events")
    return train, test

def compute_item_popularity(df, window_days=7):
    print(f"\n Computing item popularity over last {window_days} days ...")
    max_ts = df['timestamp'].max()
    cutoff = max_ts - timedelta(days=window_days)
    recent = df[df['timestamp'] >= cutoff]
    pop = recent['item_id'].value_counts()

    print(f"• Popularity computed on {len(recent):,} events")
    print(f"• Top items:")
    print(pop.head(5))
    return pop.to_dict()

if __name__ == "__main__":
    print("\n Starting YooChoose preprocessing\n" + "-"*50)

    df = load_yoochoose_clicks("data/yoochoose-clicks.dat")
    df = filter_and_build_sessions(df)

    train, test = make_train_test(df, test_last_days=1)
    pop = compute_item_popularity(train, window_days=7)

    # Save everything
    train.to_parquet(DATA_DIR / "train.parquet")
    test.to_parquet(DATA_DIR / "test.parquet")
    pd.Series(pop).to_csv(DATA_DIR / "popularity_7d.csv")

    print("\nSaved:")
    print("• data/train.parquet")
    print("• data/test.parquet")
    print("• data/popularity_7d.csv") 
