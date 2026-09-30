# Fake Job Posting Detector: Project Report

**Student:** Raj Verma  **Reg. No.:** 25MIM10221
**Programme:** Integrated M.Tech Artificial Intelligence, VIT Bhopal University
**Platform:** VITyarthi  **Date:** September 2026

---

## Abstract

Fraudulent job postings harm job seekers financially and emotionally. This project builds a Fake Job Posting Detector that combines TF-IDF text features with nine rule-based red-flag features and classifies postings using class-balanced logistic regression. On a stratified 20% test split of a synthetic dataset, it reaches 94.4% accuracy, 87.5% precision, 84.8% recall and an F1 score of 0.862, outperforming a rule-only baseline (F1 0.800) and a majority-class baseline. Each prediction includes a probability, a verdict and an explanation of the red flags and terms behind it. Because the data is synthetic, results are optimistic and are discussed with that caveat.

## 1. Introduction

### 1.1 Motivation
Job scams are common on job portals and messaging apps. Typical patterns include requests for upfront fees, contact only through WhatsApp or Telegram, personal email addresses, and unrealistic pay for easy work. Students and freshers are frequent targets. An automated first-pass check can warn applicants before they lose money or share sensitive information.

### 1.2 Problem statement
Classify a job posting as genuine or fraudulent from its text, output a fraud probability, and justify the decision.

### 1.3 Objectives
1. Build an NLP classifier for fake job postings.
2. Add interpretable rule-based red-flag features.
3. Handle class imbalance appropriately.
4. Evaluate with metrics suited to imbalanced data.
5. Provide explainable, tested and reproducible software.

## 2. Background

**Text classification** turns documents into numeric vectors and learns a decision boundary between classes.

**TF-IDF** weights a word by its frequency in a document, discounted by how common it is across all documents, highlighting distinctive words. Bigrams (two-word phrases) capture phrases such as "registration fee".

**Logistic regression** models the probability of the positive class as `p = 1 / (1 + e^-(w.x + b))`. It is fast, works well on sparse text, and its weights are interpretable, which supports explanations.

**Class imbalance.** Fraud is a minority class (about 21% here). Accuracy alone is misleading: a model that always says "genuine" scores 79.4% accuracy while catching nothing. Precision, recall and F1 give a fuller picture:
- Precision = TP / (TP + FP): of postings flagged as fake, how many really are.
- Recall = TP / (TP + FN): of all fake postings, how many were caught.
- F1 = harmonic mean of precision and recall.

## 3. System Design

### 3.1 Architecture

```
 Job posting (title, profile, description, requirements, salary)
        │
        ├─► clean_text ─► TF-IDF (1-2 grams, max 5000 features) ─┐
        │                                                         ├─► Logistic Regression
        └─► red_flags (9 binary rules) ──────────────────────────┘        │
                                                                          ▼
                                                   probability + verdict + red flags + top terms
```

### 3.2 Modules

| Module | Responsibility |
|---|---|
| `features.py` | Text cleaning and the 9 red-flag rules |
| `model.py` | Dataset loading and validation, training, evaluation, prediction and explanation, save/load |
| `cli.py` | Commands: train, evaluate, predict, interactive |
| `data/generate_dataset.py` | Reproducible synthetic dataset |

### 3.3 Dataset
800 postings (635 genuine, 165 fraudulent) with columns `title, company_profile, description, requirements, salary_range, fraudulent`. The data is **synthetic** (seed 42):
- Genuine postings combine 3-5 generic responsibilities, 2-4 requirements, an optional company profile and salary. About 35% include shared phrases such as "work from home options available" or "immediate joining preferred".
- Fraudulent postings contain 1-4 scam sentences from a bank of ten (fees, free email, WhatsApp/Telegram, guaranteed income, urgency, and so on), mixed with 0-2 ordinary sentences, and often lack a company profile.
- 3% of labels are flipped to imitate labelling noise, which caps achievable accuracy at roughly 97%.

### 3.4 Red-flag features

| Feature | Rule |
|---|---|
| upfront_fee | mentions registration/processing/training fee, security deposit, "refundable" |
| messaging_app | mentions WhatsApp or Telegram |
| free_email | contains an address at gmail, yahoo, outlook, hotmail or rediffmail |
| unrealistic_pay | "earn $N per day", "guaranteed income", "get paid weekly" |
| no_experience | "no experience", "anyone can apply", "no interview" |
| urgency | "urgent", "limited seats", "apply now", "hurry" |
| excessive_caps | more than 30% of letters are uppercase |
| missing_company_profile | company profile is empty |
| very_short | description has fewer than 15 words |

## 4. Methodology

1. **Preprocessing:** the five text fields are concatenated, lower-cased, URLs replaced by a token, digits and most punctuation removed.
2. **Vectorisation:** `TfidfVectorizer` with unigrams and bigrams, English stop-words removed, minimum document frequency 2, at most 5,000 features.
3. **Feature fusion:** the sparse TF-IDF matrix is horizontally stacked with the 9 binary red-flag columns.
4. **Classifier:** `LogisticRegression(C=3, class_weight="balanced")`. Balanced weights raise the cost of missing a fake posting.
5. **Explanation:** the triggered red flags are listed, and the top TF-IDF terms are ranked by their contribution (feature value times coefficient) toward the fraud class. Terms are shown only when the probability is 40% or higher.
6. **Verdict:** probability ≥ 0.70 is high risk, 0.40-0.70 suspicious, below 0.40 likely genuine.
7. **Evaluation:** stratified 80/20 split (640 train, 160 test, seed 42), compared with:
   - *Rule baseline:* predict fraud when two or more red flags fire.
   - *Majority baseline:* always predict genuine.

## 5. Implementation

- **Stack:** Python 3.10+, pandas, SciPy, scikit-learn, joblib.
- **Interface:** `argparse` CLI with input validation and non-zero exit codes for errors. The model auto-trains on first `predict`.
- **Persistence:** the trained pipeline is saved with joblib to `models/` (git-ignored).
- **Testing:** 15 pytest tests cover dataset validation, text cleaning, each red-flag rule, a clean posting raising no flags, obvious scam scoring at least 0.7, obvious genuine posting scoring under 0.4, beating the majority baseline, save/load round-trip, untrained-model and missing-file errors, bad-dataset rejection and CLI behaviour.
- **CI:** GitHub Actions runs the tests on Python 3.10, 3.11 and 3.12.

## 6. Results

### 6.1 Metrics (test set: 160 postings, 33 fraudulent)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression (proposed) | 0.944 | 0.875 | 0.848 | 0.862 |
| Rule-only baseline | 0.925 | 0.889 | 0.727 | 0.800 |
| Majority baseline | 0.794 | 0.000 | 0.000 | 0.000 |

Confusion matrix (proposed model): TN = 123, FP = 4, FN = 5, TP = 28.

### 6.2 Sample predictions

| Input | Probability | Verdict |
|---|---|---|
| "Online Typist": urgent hiring, $500 per day, no experience, refundable fee, WhatsApp | 100.0% | High risk, 6 red flags |
| "Data Analyst" at a described finance company, routine responsibilities | 12.3% | Likely genuine, no red flags |

### 6.3 Discussion
- The model catches about 85% of fake postings (recall) and about 87% of its fake alerts are correct (precision).
- Compared with the rule-only baseline, the classifier gains 12 points of recall for a small loss of precision. Learned text patterns catch scams that trigger only one or no rule.
- The rule baseline is already strong (F1 0.800). This is expected: the synthetic scam sentences are built from the same kinds of phrases the rules look for. On real postings, where scammers vary their wording, the gap between rules and a learned model could be larger or smaller.
- With 3% label noise the ceiling is around 97% accuracy, so 94.4% is near the practical limit for this dataset.
- The 160-example test set contains only 33 fraud cases, so each error moves recall by about 3 points. The results carry substantial uncertainty and should not be over-interpreted.

## 7. Limitations

1. **Synthetic data:** real scams are more varied and subtle, so real-world performance will likely be lower.
2. **Single split:** no cross-validation or confidence intervals.
3. **Language:** only English text is handled.
4. **Text only:** the system does not check the company's website, email domain or reputation.
5. **Adaptive adversaries:** scammers can rewrite ads to avoid known phrases.
6. **Fixed thresholds:** the 0.40 and 0.70 cut-offs were chosen by judgement, not tuned.

## 8. Future Work

- Train and test on the real EMSCAD Fake Job Postings dataset.
- Compare with Naive Bayes, SVM, Random Forest and transformer models, using k-fold cross-validation.
- Tune the decision threshold for the desired precision/recall trade-off.
- Add domain, WHOIS and blocklist checks, and multilingual support.
- Build a web interface or browser extension.

## 9. Conclusion

The Fake Job Posting Detector shows that combining simple NLP features with interpretable red-flag rules gives a fast, explainable screening tool. It outperforms both a rule-only and a majority baseline on the synthetic test set, and it tells users why a posting looks suspicious. The main caveat is the synthetic dataset; validating on real postings is the most important next step.

## 10. References

1. Vidros, S., Kolias, C., Kambourakis, G., Akoglu, L., "Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset," *Future Internet*, 2017 (EMSCAD dataset).
2. Manning, C., Raghavan, P., Schütze, H., *Introduction to Information Retrieval*, Cambridge University Press, 2008.
3. Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 2011.
4. scikit-learn documentation: https://scikit-learn.org
5. pandas documentation: https://pandas.pydata.org
