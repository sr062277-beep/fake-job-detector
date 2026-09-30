import csv
import subprocess
import sys

import pytest

from job_detector import JobDetector, clean_text, red_flags
from job_detector.cli import main
from job_detector.model import load_dataset

SCAM = dict(
    title="Online Typist",
    description=("URGENT HIRING! Earn $500 per day from home. No experience needed. "
                 "Pay a refundable registration fee of $50. Send resume to hr.jobs@gmail.com "
                 "or WhatsApp us."),
)
GENUINE = dict(
    title="Data Analyst",
    company_profile="Vertex Analytics is a growing finance company with 1200 employees.",
    description=("Analyse data to identify trends and support business decisions. "
                 "Prepare weekly reports and present findings to senior management. "
                 "Collaborate with cross-functional teams to deliver quarterly goals."),
    requirements="Bachelor's degree in a relevant field. Proficiency in SQL.",
    salary_range="60000-110000",
)


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def model(df):
    return JobDetector().fit(df)


def test_dataset_valid(df):
    assert len(df) == 800
    assert set(df["fraudulent"].unique()) == {0, 1}


def test_generator_reproducible(tmp_path):
    out = subprocess.run([sys.executable, "data/generate_dataset.py"],
                         capture_output=True, text=True, check=True).stdout
    assert "800 postings" in out


def test_clean_text():
    assert clean_text("Visit https://x.com NOW!! $500") == "visit url now"
    assert clean_text(None) == ""


def test_red_flags_detect_scam():
    f = red_flags(SCAM["title"], "", SCAM["description"])
    for k in ("upfront_fee", "free_email", "messaging_app", "unrealistic_pay",
              "no_experience", "urgency", "missing_company_profile"):
        assert f[k] == 1, k


def test_red_flags_clean_posting():
    f = red_flags(GENUINE["title"], GENUINE["company_profile"], GENUINE["description"],
                  GENUINE["requirements"], GENUINE["salary_range"])
    assert sum(f.values()) == 0


def test_excessive_caps_and_short():
    f = red_flags("JOB", "x", "EARN MONEY NOW")
    assert f["excessive_caps"] == 1 and f["very_short"] == 1


def test_predict_scam_high(model):
    r = model.predict(**SCAM)
    assert r["fraud_probability"] >= 0.7
    assert r["red_flags"]


def test_predict_genuine_low(model):
    r = model.predict(**GENUINE)
    assert r["fraud_probability"] < 0.4
    assert r["verdict"] == "LIKELY GENUINE"


def test_evaluate_beats_majority(df):
    r = JobDetector().evaluate(df)
    assert r["model"]["f1"] > r["majority_baseline"]["f1"]
    assert r["model"]["recall"] > 0.5
    assert r["n_train"] + r["n_test"] == len(df)


def test_predict_untrained_raises():
    with pytest.raises(RuntimeError):
        JobDetector().predict(description="x")


def test_save_load_roundtrip(model, tmp_path):
    p = model.save(tmp_path / "m.joblib")
    loaded = JobDetector.load(p)
    assert loaded.predict(**SCAM)["fraud_probability"] == model.predict(**SCAM)["fraud_probability"]


def test_load_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        JobDetector.load(tmp_path / "nope.joblib")


def test_bad_dataset_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows([["title", "fraudulent"], ["x", 1]])
    with pytest.raises(ValueError):
        load_dataset(p)


def test_cli_train_and_predict(tmp_path, capsys):
    mp = str(tmp_path / "m.joblib")
    assert main(["--model", mp, "train"]) == 0
    assert main(["--model", mp, "predict", "--title", SCAM["title"],
                 "--text", SCAM["description"]]) == 0
    assert "HIGH RISK" in capsys.readouterr().out


def test_cli_predict_needs_input(tmp_path, capsys):
    assert main(["--model", str(tmp_path / "m.joblib"), "predict"]) == 1
