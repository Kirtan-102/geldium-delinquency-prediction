"""
Model training and evaluation.

Responsibilities:
- Build transparent LogisticRegression baseline pipeline (preferred for production transparency)
- Optionally train XGBoost (performance) with SMOTE oversampling
- Produce evaluation metrics including precision/recall/F1/ROC-AUC/PR-AUC
- Persist final model and an auditable scored test CSV
- Produce explainability artifacts (coefficients, feature importances, optional SHAP)
"""
from pathlib import Path
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
# imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# SHAP optional
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

def get_feature_sets(df: pd.DataFrame) -> Tuple[List[str], List[str], pd.Series]:
    numeric_features = [
        'Credit_Utilization', 'Missed_Payments', 'Income', 'Debt_to_Income_Ratio',
        'Account_Tenure', 'Credit_Score', 'Loan_Balance', 'Age',
        'count_missed', 'count_late', 'recent_missed', 'recent_late',
        'loan_balance_missing', 'income_missing'
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in ['Employment_Status', 'Credit_Card_Type', 'Location'] if c in df.columns]
    y = df['Delinquent_Account']
    return numeric_features, categorical_features, y

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')
    return preprocessor

def train_logistic(preprocessor, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    # Balanced class weights to address class imbalance in a transparent way
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgboost_with_smote(preprocessor, X_train: pd.DataFrame, y_train: pd.Series) -> ImbPipeline:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost not available; install xgboost to use this function.")
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', XGBClassifier(n_estimators=200, max_depth=4, use_label_encoder=False, eval_metric='logloss',
                              random_state=RANDOM_STATE))
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

def get_preprocessed_feature_names(preprocessor) -> List[str]:
    """
    Return feature names after ColumnTransformer + OneHotEncoder
    """
    feature_names = []
    # numeric columns
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(cols)
        if name == 'cat':
            ohe = trans.named_steps['onehot']
            try:
                ohe_cols = ohe.get_feature_names_out(cols)
            except Exception:
                ohe_cols = ohe.get_feature_names(cols)
            feature_names.extend(list(ohe_cols))
    return feature_names

def persist_pipeline(pipeline, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved pipeline to %s", path)

def persist_audit_scores(pipeline, X_test: pd.DataFrame, y_test: pd.Series, ids: pd.Series, out_csv: str):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df = X_test.copy()
    df['Customer_ID'] = ids.values
    df['true_label'] = y_test.values
    df['pred_proba'] = pipeline.predict_proba(X_test)[:, 1]
    df['pred_label'] = pipeline.predict(X_test)
    df.to_csv(out_csv, index=False)
    logger.info("Saved audit CSV to %s", out_csv)

def explain_logistic(pipeline, preprocessor):
    """
    Extract and return coefficient table for the logistic regression model.
    """
    clf = pipeline.named_steps['clf']
    coef = clf.coef_[0]
    feature_names = get_preprocessed_feature_names(preprocessor)
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
    coef_df = coef_df.sort_values(by='coefficient', key=lambda x: np.abs(x), ascending=False)
    return coef_df

def explain_xgb_shap(xgb_pipeline, X_sample: pd.DataFrame, preprocessor):
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available. Skip SHAP explanation.")
        return None
    # Transform sample
    pre = xgb_pipeline.named_steps['preprocessor']
    clf = xgb_pipeline.named_steps['clf']
    X_trans = pre.transform(X_sample)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    feature_names = get_preprocessed_feature_names(pre)
    # returns objects for downstream plotting
    return dict(shap_values=shap_values, X_trans=X_trans, feature_names=feature_names)

def run_training_pipeline(processed_csv: str,
                          out_model_path: str = "models/risk_model_logistic.joblib",
                          out_xgb_path: str = "models/risk_model_xgb.joblib",
                          audit_csv: str = "models/scored_test_set_audit.csv",
                          prefer_transparency: bool = True):
    df = pd.read_csv(processed_csv)
    numeric_features, categorical_features, y = get_feature_sets(df)
    X = df[numeric_features + categorical_features]
    ids = df['Customer_ID'] if 'Customer_ID' in df.columns else pd.Series(range(len(df)), name='row_id')
    # stratified split to keep target distribution in train/test
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Train logistic baseline
    logger.info("Training logistic regression baseline...")
    log_pipe = train_logistic(preprocessor, X_train, y_train)
    log_metrics = evaluate_model(log_pipe, X_test, y_test)
    logger.info("Logistic performance: Acc %.3f, Precision %.3f, Recall %.3f, F1 %.3f, ROC-AUC %.3f, PR-AUC %.3f",
                log_metrics['accuracy'], log_metrics['precision'], log_metrics['recall'], log_metrics['f1'],
                log_metrics['roc_auc'], log_metrics['pr_auc'])
    coef_df = explain_logistic(log_pipe, preprocessor)
    coef_df.to_csv("models/logistic_coefficients.csv", index=False)
    logger.info("Saved logistic coefficients to models/logistic_coefficients.csv")

    # Persist logistic model
    persist_pipeline(log_pipe, out_model_path)

    # Persist audit CSV for model governance
    persist_audit_scores(log_pipe, X_test, y_test, id_test, audit_csv)

    # Optionally train XGBoost + SMOTE if available and desired
    xgb_pipe = None
    if XGBOOST_AVAILABLE:
        logger.info("Training XGBoost with SMOTE to improve recall (if needed)...")
        xgb_pipe = train_xgboost_with_smote(preprocessor, X_train, y_train)
        xgb_metrics = evaluate_model(xgb_pipe, X_test, y_test)
        logger.info("XGBoost performance: Acc %.3f, Precision %.3f, Recall %.3f, F1 %.3f, ROC-AUC %.3f, PR-AUC %.3f",
                    xgb_metrics['accuracy'], xgb_metrics['precision'], xgb_metrics['recall'], xgb_metrics['f1'],
                    xgb_metrics['roc_auc'], xgb_metrics['pr_auc'])
        # Save feature importances
        names = get_preprocessed_feature_names(preprocessor)
        try:
            importances = xgb_pipe.named_steps['clf'].feature_importances_
            fi_df = pd.DataFrame({'feature': names, 'importance': importances}).sort_values('importance', ascending=False)
            fi_df.to_csv("models/xgb_feature_importances.csv", index=False)
            logger.info("Saved XGBoost importances to models/xgb_feature_importances.csv")
        except Exception:
            logger.warning("Could not extract feature importances from XGBoost.")
        persist_pipeline(xgb_pipe, out_xgb_path)
        # Optional SHAP explanation on a sample
        if SHAP_AVAILABLE:
            sample = X_test.sample(min(200, len(X_test)), random_state=RANDOM_STATE)
            shap_out = explain_xgb_shap(xgb_pipe, sample, preprocessor)
            # Leave saving/plotting to interactive sessions (Jupyter)
    else:
        logger.info("XGBoost not installed; skipping XGBoost + SMOTE training.")

    # Model selection suggestion: by default select logistic (transparent).
    final_model_path = out_model_path
    logger.info("Training complete. Final model (by default) persisted at: %s", final_model_path)

    return {
        'logistic_metrics': log_metrics,
        'logistic_coef': coef_df,
        'xgb_metrics': xgb_metrics if XGBOOST_AVAILABLE else None,
        'xgb_pipe': xgb_pipe
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/processed.csv", help="Path to processed CSV")
    parser.add_argument("--model-out", default="models/risk_model_logistic.joblib")
    parser.add_argument("--xgb-out", default="models/risk_model_xgb.joblib")
    parser.add_argument("--audit-csv", default="models/scored_test_set_audit.csv")
    args = parser.parse_args()
    run_training_pipeline(args.processed, args.model_out, args.xgb_out, args.audit_csv)
