# 🕵️ Fake Job Posting Detector

[![CI](https://github.com/YOUR-USERNAME/fake-job-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/fake-job-detector/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine-learning tool that reads a job posting and estimates how likely it is to be a **scam**. It combines **NLP (TF-IDF)**, **rule-based red-flag detection** and a **logistic regression** classifier, and explains *why* a posting looks suspicious.


**Course project:** VITyarthi

---

## Features

- **Fraud probability** (0-100%) with a plain-English verdict: *Likely genuine*, *Suspicious*, or *High risk*
- **Explainable output:** lists red flags (upfront fees, WhatsApp/Telegram contact, free-email recruiters, unrealistic pay, urgency, and more) and the most suspicious words
- **Hybrid model:** TF-IDF text features plus 9 hand-built red-flag features
- **Evaluation** with accuracy, precision, recall, F1 and a confusion matrix, compared against two baselines
- **Interactive mode** to paste postings and get instant verdicts
- Auto-trains on first use, so `predict` works right after cloning
- 15 unit tests and a GitHub Actions CI workflow

## How it works

```
Job posting ─► clean text ─► TF-IDF (1-2 word n-grams) ─┐
            └► 9 red-flag rules (regex + heuristics) ───┴► Logistic Regression ─► probability + explanation
```

| Red flag | Example trigger |
|---|---|
| Upfront fee | "registration fee", "security deposit" |
| Messaging-app contact | "WhatsApp", "Telegram" |
| Free email | recruiter uses `@gmail.com` |
| Unrealistic pay | "earn $500 per day", "guaranteed income" |
| No experience needed | "no experience", "anyone can apply" |
| Urgency | "URGENT", "limited seats" |
| Excessive caps | more than 30% capital letters |
| Missing company profile | empty profile field |
| Very short description | under 15 words |

Verdict thresholds: probability ≥ 70% is **High risk**, 40-70% is **Suspicious**, below 40% is **Likely genuine**.

## Project structure

```
fake-job-detector/
├── job_detector/
│   ├── __init__.py
│   ├── __main__.py        # python -m job_detector
│   ├── cli.py             # command-line interface
│   ├── features.py        # text cleaning and red-flag rules
│   └── model.py           # training, evaluation, prediction, save/load
├── data/
│   ├── job_postings.csv       # 800 synthetic postings (~21% fraudulent)
│   └── generate_dataset.py    # reproducible generator (seed 42)
├── tests/test_detector.py
├── .github/workflows/ci.yml
├── PROJECT_REPORT.md
├── PROJECT_DESCRIPTION.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/fake-job-detector.git
cd fake-job-detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Usage

```bash
python -m job_detector train                # train and save the model
python -m job_detector evaluate             # metrics vs baselines
python -m job_detector predict --title "Online Typist" --text "Earn $500 per day. Pay a registration fee. WhatsApp us."
python -m job_detector predict --file posting.txt
python -m job_detector interactive
```

### Sample output

```
$ python -m job_detector predict --title "Online Typist" --text "URGENT HIRING! Earn $500 per day
  from home. No experience needed. Pay a refundable registration fee. WhatsApp us."

Fraud probability : 100.0%
Verdict           : HIGH RISK - likely fake
Red flags:
  - Asks for a fee or deposit before you start
  - Recruiter contact only via WhatsApp/Telegram
  - Promises unrealistic or guaranteed earnings
  - Claims no experience or interview is needed
  - Uses pressure or urgency language
  - No company profile provided
Suspicious terms  : experience needed, needed, hiring, pay, online typist
```

```
$ python -m job_detector predict --title "Data Analyst" --company-profile "Vertex Analytics is a growing
  finance company." --text "Analyse data to identify trends and support business decisions..."

Fraud probability : 12.3%
Verdict           : LIKELY GENUINE
Red flags         : none detected
```

### Use as a library

```python
from job_detector import JobDetector
from job_detector.model import load_dataset

model = JobDetector().fit(load_dataset())
print(model.predict(title="Home Based Agent",
                    description="Earn $800 per day, no experience needed"))
```

## Evaluation results

`python -m job_detector evaluate` (stratified 80/20 split, seed 42; 640 train, 160 test):

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic regression (this project)** | **0.944** | 0.875 | **0.848** | **0.862** |
| Rule-only baseline (2+ red flags) | 0.925 | 0.889 | 0.727 | 0.800 |
| Always predict "genuine" | 0.794 | 0.000 | 0.000 | 0.000 |

Confusion matrix: TN = 123, FP = 4, FN = 5, TP = 28.

The model beats both baselines on F1 and finds about 85% of fake postings. **These numbers come from synthetic data and are optimistic**; see the dataset note and the report's limitations.

## Running tests

```bash
pytest -q
```

## Dataset note

The dataset is **synthetic**, generated by `data/generate_dataset.py`. Fraudulent postings mix 1-4 common scam phrases with ordinary job-ad sentences; genuine postings sometimes reuse phrases like "work from home" so the classes overlap; 3% of labels are flipped as noise. To use real data (for example the public EMSCAD "Fake Job Postings" dataset on Kaggle), provide a CSV with the columns `title, company_profile, description, requirements, salary_range, fraudulent` and pass `--data your_file.csv`.

## Disclaimer

This is an educational project, not a guarantee. A "Likely genuine" verdict does **not** prove a job is real. Always verify the company independently, and never pay money to get a job.

## Uploading this project to GitHub (from the terminal)

```bash
cd fake-job-detector
git init
git add .
git commit -m "Initial commit: Fake Job Posting Detector"
git branch -M main
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

If the remote repo already has files, run `git pull origin main --allow-unrelated-histories` before pushing. Afterwards, replace `YOUR-USERNAME` in the badge links with the real GitHub username.

## Future improvements

- Train on the real EMSCAD dataset and compare against Naive Bayes, SVM and Random Forest
- Cross-validation and threshold tuning for a better precision/recall balance
- Check company domains and email addresses against WHOIS or blocklists
- Web interface (Streamlit/Flask) or a browser extension

## License

MIT - see [LICENSE](LICENSE).
