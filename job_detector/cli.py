"""Command-line interface.

Examples:
  python -m job_detector train
  python -m job_detector evaluate
  python -m job_detector predict --title "Online Typist" --text "Earn $500 per day, pay a registration fee"
  python -m job_detector predict --file posting.txt
  python -m job_detector interactive
"""
import argparse
import sys
from pathlib import Path

from .model import DEFAULT_MODEL, JobDetector, load_dataset


def _show(result):
    print(f"\nFraud probability : {result['fraud_probability']:.1%}")
    print(f"Verdict           : {result['verdict']}")
    if result["red_flags"]:
        print("Red flags:")
        for f in result["red_flags"]:
            print(f"  - {f}")
    else:
        print("Red flags         : none detected")
    if result["suspicious_terms"]:
        print("Suspicious terms  : " + ", ".join(result["suspicious_terms"]))


def _get_model(args):
    path = args.model or DEFAULT_MODEL
    if not Path(path).exists():
        print("No trained model found, training one now...")
        m = JobDetector().fit(load_dataset(args.data))
        m.save(path)
        return m
    return JobDetector.load(path)


def build_parser():
    p = argparse.ArgumentParser(prog="job_detector", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", help="path to a job postings CSV")
    p.add_argument("--model", help="path to save/load the trained model")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="train on the full dataset and save the model")
    sub.add_parser("evaluate", help="80/20 split evaluation with baselines")
    pr = sub.add_parser("predict", help="check a single job posting")
    pr.add_argument("--title", default="")
    pr.add_argument("--text", default="", help="job description text")
    pr.add_argument("--company-profile", default="")
    pr.add_argument("--requirements", default="")
    pr.add_argument("--salary", default="")
    pr.add_argument("--file", help="text file containing the job description")
    sub.add_parser("interactive", help="paste postings and get instant verdicts")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "train":
        df = load_dataset(args.data)
        path = JobDetector().fit(df).save(args.model)
        print(f"Trained on {len(df)} postings. Model saved to {path}")
        return 0

    if args.command == "evaluate":
        r = JobDetector().evaluate(load_dataset(args.data))
        print(f"Train: {r['n_train']} postings | Test: {r['n_test']} postings\n")
        print(f"{'':20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
        for name in ("model", "rule_baseline", "majority_baseline"):
            s = r[name]
            print(f"{name:20}{s['accuracy']:>10.3f}{s['precision']:>11.3f}"
                  f"{s['recall']:>9.3f}{s['f1']:>8.3f}")
        (tn, fp), (fn, tp) = r["confusion_matrix"]
        print(f"\nConfusion matrix (model): TN={tn} FP={fp} FN={fn} TP={tp}")
        return 0

    if args.command == "predict":
        text = args.text
        if args.file:
            try:
                text = Path(args.file).read_text(encoding="utf-8")
            except OSError as e:
                print(f"Error: cannot read file: {e}", file=sys.stderr)
                return 1
        if not (args.title or text):
            print("Error: provide --text, --file or --title", file=sys.stderr)
            return 1
        _show(_get_model(args).predict(
            title=args.title, description=text, company_profile=args.company_profile,
            requirements=args.requirements, salary_range=args.salary))
        return 0

    if args.command == "interactive":
        model = _get_model(args)
        print("Paste a job description and press Enter (type 'quit' to exit).")
        while True:
            text = input("\nJob text> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                return 0
            if text:
                _show(model.predict(description=text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
