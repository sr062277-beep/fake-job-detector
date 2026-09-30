# Project Description

**Title:** Fake Job Posting Detector (NLP and Machine Learning)
**Student:** Raj Verma | **Reg. No.:** 25MIM10221
**Programme:** Integrated M.Tech Artificial Intelligence, VIT Bhopal University
**Platform:** VITyarthi

## Overview

Online job scams trick job seekers into paying fake "registration fees", sharing personal documents or working for nothing. Fraudulent ads are easy to post and hard for a busy applicant to spot. This project builds a system that reads a job posting and estimates the probability that it is fake, then explains its reasoning in plain language. It is written in Python with scikit-learn and used through a command-line interface.

## Problem Statement

Given the text of a job posting, classify it as genuine or fraudulent and tell the user which warning signs triggered the decision.

## Objectives

1. Build a text classifier that separates fraudulent from genuine postings.
2. Add rule-based red-flag detection for well-known scam patterns.
3. Combine text and rule features in a single, explainable model.
4. Report a fraud probability and a clear verdict instead of a bare label.
5. Evaluate with precision, recall and F1, since fraud data is imbalanced, and compare against baselines.
6. Deliver tested, documented, GitHub-ready code.

## Approach

- **Data:** 800 synthetic postings (165 fraudulent, about 21%) with title, company profile, description, requirements and salary fields.
- **Preprocessing:** lower-casing, URL and symbol removal, whitespace normalisation.
- **Features:** TF-IDF on 1-2 word n-grams (up to 5,000) plus 9 red-flag indicators.
- **Model:** logistic regression with balanced class weights to handle the imbalance.
- **Explainability:** triggered red flags and the words that contributed most to the fraud score.
- **Evaluation:** stratified 80/20 split, compared with a rule-only baseline and an "always genuine" baseline.

## Tools and Technologies

Python 3.10+, pandas, NumPy, SciPy, scikit-learn, joblib, pytest, Git and GitHub Actions.

## Outcomes

- Working CLI with commands `train`, `evaluate`, `predict` and `interactive`.
- Test results: accuracy 94.4%, precision 87.5%, recall 84.8%, F1 0.862, beating the rule-only baseline (F1 0.800).
- Human-readable explanations for every prediction.
- 15 automated tests passing on Python 3.10, 3.11 and 3.12.

## Limitations

The dataset is synthetic, so the scores are optimistic and real-world performance would likely be lower. The system judges wording only; it does not verify companies, websites or email domains. It should be treated as a screening aid, not proof that a job is genuine.
