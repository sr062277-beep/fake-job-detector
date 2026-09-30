"""Training, evaluation and prediction."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

from .features import (FLAG_DESCRIPTIONS, FLAG_NAMES, clean_text, combine_fields,
                       red_flags)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "job_postings.csv"
DEFAULT_MODEL = ROOT / "models" / "job_detector.joblib"
TEXT_COLS = ["title", "company_profile", "description", "requirements", "salary_range"]


def load_dataset(path=None):
    df = pd.read_csv(path or DEFAULT_DATA).fillna("")
    for c in TEXT_COLS + ["fraudulent"]:
        if c not in df.columns:
            raise ValueError(f"Dataset is missing column '{c}'")
    if not set(df["fraudulent"].unique()) <= {0, 1}:
        raise ValueError("'fraudulent' must contain only 0 and 1")
    return df


def risk_level(prob):
    if prob >= 0.70:
        return "HIGH RISK - likely fake"
    if prob >= 0.40:
        return "SUSPICIOUS - verify before applying"
    return "LIKELY GENUINE"


class JobDetector:
    def __init__(self, C=3.0):
        self.C = C
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, stop_words="english", max_features=5000
        )
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        self.fitted = False

    # ---- feature building -------------------------------------------------
    def _flag_matrix(self, df):
        rows = [list(red_flags(*[r[c] for c in TEXT_COLS]).values())
                for _, r in df.iterrows()]
        return csr_matrix(np.array(rows, dtype=float))

    def _texts(self, df):
        return [clean_text(combine_fields(*[r[c] for c in TEXT_COLS]))
                for _, r in df.iterrows()]

    def _features(self, df, fit=False):
        texts = self._texts(df)
        tfidf = self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)
        return hstack([tfidf, self._flag_matrix(df)]).tocsr()

    # ---- training / evaluation -------------------------------------------
    def fit(self, df):
        self.clf.fit(self._features(df, fit=True), df["fraudulent"].values)
        self.fitted = True
        return self

    def predict_proba_df(self, df):
        return self.clf.predict_proba(self._features(df))[:, 1]

    @staticmethod
    def split(df, test_size=0.2, seed=42):
        return train_test_split(df, test_size=test_size, random_state=seed,
                                stratify=df["fraudulent"])

    def evaluate(self, df, seed=42, threshold=0.5):
        """Train on 80%, test on 20%. Also compares with simple baselines."""
        train, test = self.split(df, seed=seed)
        self.fit(train)
        y = test["fraudulent"].values
        pred = (self.predict_proba_df(test) >= threshold).astype(int)
        rule_pred = (self._flag_matrix(test).sum(axis=1).A1 >= 2).astype(int)

        def scores(p):
            return {
                "accuracy": accuracy_score(y, p),
                "precision": precision_score(y, p, zero_division=0),
                "recall": recall_score(y, p, zero_division=0),
                "f1": f1_score(y, p, zero_division=0),
            }

        return {
            "model": scores(pred),
            "rule_baseline": scores(rule_pred),
            "majority_baseline": scores(np.zeros_like(y)),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "n_train": len(train),
            "n_test": len(test),
        }

    # ---- prediction -------------------------------------------------------
    def predict(self, title="", description="", company_profile="", requirements="",
                salary_range="", top_terms=5):
        if not self.fitted:
            raise RuntimeError("Model is not trained")
        df = pd.DataFrame([{"title": title, "company_profile": company_profile,
                            "description": description, "requirements": requirements,
                            "salary_range": salary_range}])
        x = self._features(df)
        prob = float(self.clf.predict_proba(x)[0, 1])

        names = list(self.vectorizer.get_feature_names_out()) + FLAG_NAMES
        contrib = x.multiply(self.clf.coef_[0]).tocoo()
        n_text = len(names) - len(FLAG_NAMES)
        terms = sorted(
            ((names[j], v) for j, v in zip(contrib.col, contrib.data) if j < n_text and v > 0),
            key=lambda t: -t[1],
        )[:top_terms]

        if prob < 0.40:
            terms = []  # only explain terms when the posting looks risky
        flags = red_flags(title, company_profile, description, requirements, salary_range)
        return {
            "fraud_probability": round(prob, 3),
            "verdict": risk_level(prob),
            "red_flags": [FLAG_DESCRIPTIONS[k] for k, v in flags.items() if v],
            "suspicious_terms": [t for t, _ in terms],
        }

    # ---- persistence ------------------------------------------------------
    def save(self, path=None):
        path = Path(path or DEFAULT_MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None):
        path = Path(path or DEFAULT_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run: python -m job_detector train")
        return joblib.load(path)
