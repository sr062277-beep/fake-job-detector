"""Generate a reproducible synthetic dataset of job postings.

Output: data/job_postings.csv  (800 rows, ~20% fraudulent)

Fraudulent postings contain 1-4 typical scam phrases (upfront fees, free-email
contacts, messaging-app contacts, unrealistic pay, ...) mixed with ordinary
job-ad sentences. Genuine postings sometimes reuse phrases such as "work from
home" or "immediate joining" so the classes overlap. About 3% of labels are
flipped to mimic labelling noise.

Run:  python data/generate_dataset.py
"""
import csv
import random
from pathlib import Path

SEED = 42
N = 800
FRAUD_RATE = 0.20
LABEL_NOISE = 0.03
HERE = Path(__file__).parent

LEGIT_TITLES = [
    "Software Engineer", "Data Analyst", "Marketing Executive", "Accountant",
    "Mechanical Engineer", "HR Coordinator", "Sales Manager", "Graphic Designer",
    "Customer Support Associate", "Project Manager", "Content Writer",
    "Business Analyst", "Operations Executive", "Front-end Developer",
    "Registered Nurse", "Quality Assurance Tester",
]
FRAUD_TITLES = [
    "Data Entry Operator - Work From Home", "Online Typist", "Personal Assistant",
    "Home Based Agent", "Content Writer", "Customer Support Associate",
    "Part Time Job Online", "Data Entry Clerk", "Marketing Executive",
]
COMPANIES = [
    "Nimbus Technologies", "BlueOak Financial", "Greenfield Logistics",
    "Aster Healthcare", "Pinnacle Retail", "Vertex Analytics", "Orbit Media",
    "Sunrise Manufacturing", "Lattice Software", "Harbor Consulting",
]
INDUSTRIES = ["software", "healthcare", "finance", "retail", "logistics",
              "media", "manufacturing", "consulting"]
TOOLS = ["Excel", "SQL", "Python", "Tally", "Photoshop", "Salesforce", "Jira"]

RESPONSIBILITIES = [
    "Collaborate with cross-functional teams to deliver quarterly goals.",
    "Prepare weekly reports and present findings to senior management.",
    "Maintain accurate records and ensure compliance with company policy.",
    "Coordinate with clients and resolve queries within agreed timelines.",
    "Design, develop and test features as part of an agile team.",
    "Analyse data to identify trends and support business decisions.",
    "Participate in code reviews and documentation.",
    "Manage day-to-day operations and support process improvements.",
]
LEGIT_REQ = [
    "Bachelor's degree in a relevant field.",
    "{n}+ years of relevant experience.",
    "Strong communication and teamwork skills.",
    "Proficiency in {tool}.",
    "Ability to work independently and meet deadlines.",
]
SHARED_PHRASES = [
    "This role offers a hybrid work model.",
    "Immediate joining preferred.",
    "Competitive salary and benefits.",
    "Work from home options available.",
    "Apply through our careers page.",
]
SCAM_SENTENCES = [
    "Earn ${pay} per day working from home!",
    "No experience needed, anyone can apply.",
    "A refundable registration fee of ${fee} is required to start.",
    "Send your resume to hr.{name}@gmail.com.",
    "Contact our manager on WhatsApp at +91-98XXXXXX for instant selection.",
    "URGENT HIRING! Limited seats, apply now!",
    "Guaranteed income, no interview required.",
    "Get paid weekly, just 2 hours a day.",
    "Pay a small security deposit to receive your training kit.",
    "Join our Telegram group for daily tasks and payouts.",
]


def legit_posting(rng):
    company = rng.choice(COMPANIES)
    desc = rng.sample(RESPONSIBILITIES, rng.randint(3, 5))
    if rng.random() < 0.35:
        desc.append(rng.choice(SHARED_PHRASES))
    req = [
        s.format(n=rng.randint(1, 6), tool=rng.choice(TOOLS))
        for s in rng.sample(LEGIT_REQ, rng.randint(2, 4))
    ]
    profile = (
        f"{company} is a growing {rng.choice(INDUSTRIES)} company with "
        f"{rng.randint(50, 5000)} employees."
    ) if rng.random() > 0.05 else ""
    salary = f"{rng.randrange(30, 120, 5) * 1000}-{rng.randrange(120, 200, 5) * 1000}" \
        if rng.random() < 0.6 else ""
    return {
        "title": rng.choice(LEGIT_TITLES),
        "company_profile": profile,
        "description": " ".join(desc),
        "requirements": " ".join(req),
        "salary_range": salary,
        "fraudulent": 0,
    }


def fraud_posting(rng):
    scam = [
        s.format(pay=rng.choice([300, 500, 800, 1000]),
                 fee=rng.choice([20, 50, 100]),
                 name=rng.choice(["jobs", "hiring", "career", "recruit"]) + str(rng.randint(1, 99)))
        for s in rng.sample(SCAM_SENTENCES, rng.randint(1, 4))
    ]
    neutral = rng.sample(RESPONSIBILITIES, rng.randint(0, 2))
    if rng.random() < 0.4:
        neutral.append(rng.choice(SHARED_PHRASES))
    desc = scam + neutral
    rng.shuffle(desc)
    req = "" if rng.random() < 0.5 else rng.choice(
        ["Basic computer knowledge.", "Smartphone and internet connection.",
         "Bachelor's degree preferred."])
    profile = "" if rng.random() < 0.6 else "We are a leading global company offering great opportunities."
    salary = f"{rng.randrange(3, 10) * 1000}-{rng.randrange(10, 20) * 1000} per week" \
        if rng.random() < 0.5 else ""
    return {
        "title": rng.choice(FRAUD_TITLES),
        "company_profile": profile,
        "description": " ".join(desc),
        "requirements": req,
        "salary_range": salary,
        "fraudulent": 1,
    }


def main():
    rng = random.Random(SEED)
    rows = []
    for _ in range(N):
        rows.append(fraud_posting(rng) if rng.random() < FRAUD_RATE else legit_posting(rng))
    for r in rows:
        if rng.random() < LABEL_NOISE:
            r["fraudulent"] = 1 - r["fraudulent"]
    with open(HERE / "job_postings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    frauds = sum(r["fraudulent"] for r in rows)
    print(f"Wrote {len(rows)} postings ({frauds} labelled fraudulent).")


if __name__ == "__main__":
    main()
