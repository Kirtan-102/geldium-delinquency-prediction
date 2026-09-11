"""
Minimal tests to validate preprocessing and train pipeline skeleton.
Run with: pytest -q
"""
import pandas as pd
from pathlib import Path
from src.preprocess import preprocess_core
from src.train import get_feature_sets, build_preprocessor

def test_preprocess_minimal():
    # Create a tiny synthetic sample similar to Book1 structure
    data = {
        'Customer_ID': ['C1', 'C2'],
        'Delinquent_Account': [0, 1],
        'Credit_Utilization': [0.5, 0.8],
        'Missed_Payments': [1, 3],
        'Income': [50000, 60000],
        'Debt_to_Income_Ratio': [0.3, 0.6],
        'Account_Tenure': [5, 2],
        'Credit_Score': [650, 400],
        'Loan_Balance': [10000, 20000],
        'Age': [45, 29],
        'Employment_Status': ['Employed', 'Unemployed'],
        'Credit_Card_Type': ['Standard', 'Platinum'],
        'Location': ['Chicago', 'Houston'],
        'Month_1': ['On-time', 'Missed'],
        'Month_2': ['Late', 'Late'],
        'Month_3': ['On-time', 'On-time']
    }
    df = pd.DataFrame(data)
    df2 = preprocess_core(df)
    assert 'count_missed' in df2.columns
    assert df2['count_missed'].dtype.kind in 'iu'  # integer types
    # features selection compatibility
    num_feats, cat_feats, y = get_feature_sets(df2)
    pre = build_preprocessor(num_feats, cat_feats)
    # ensure transformer fit doesn't crash on small sample
    pre.fit(df2[num_feats + cat_feats])
