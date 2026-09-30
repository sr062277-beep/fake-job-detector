"""Text cleaning and rule-based red-flag features."""
import re

FLAG_PATTERNS = {
    "upfront_fee": r"registration fee|processing fee|security deposit|training fee|refundable",
    "messaging_app": r"whatsapp|telegram",
    "free_email": r"@(gmail|yahoo|outlook|hotmail|rediffmail)\.",
    "unrealistic_pay": (
        r"earn \$?\d{3,}\s*(per|a|/)\s*(day|daily|hour)|guaranteed income|"
        r"\$\d{3,}\s*(per|a)\s*day|get paid weekly"
    ),
    "no_experience": r"no experience|without experience|anyone can apply|no interview",
    "urgency": r"urgent|limited (seats|slots|vacancies)|apply now|hurry",
}
FLAG_DESCRIPTIONS = {
    "upfront_fee": "Asks for a fee or deposit before you start",
    "messaging_app": "Recruiter contact only via WhatsApp/Telegram",
    "free_email": "Uses a free personal email (Gmail/Yahoo...) instead of a company domain",
    "unrealistic_pay": "Promises unrealistic or guaranteed earnings",
    "no_experience": "Claims no experience or interview is needed",
    "urgency": "Uses pressure or urgency language",
    "excessive_caps": "Excessive use of CAPITAL LETTERS",
    "missing_company_profile": "No company profile provided",
    "very_short": "Description is unusually short",
}
FLAG_NAMES = list(FLAG_DESCRIPTIONS)

_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in FLAG_PATTERNS.items()}


def clean_text(text):
    """Lower-case, strip URLs/digits/punctuation noise, collapse whitespace."""
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"[^a-z@\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_fields(title="", company_profile="", description="", requirements="",
                   salary_range=""):
    return " ".join(str(x or "") for x in
                    (title, company_profile, description, requirements, salary_range))


def red_flags(title="", company_profile="", description="", requirements="",
              salary_range=""):
    """Return {flag_name: 0/1} for every rule-based red flag."""
    full = combine_fields(title, company_profile, description, requirements, salary_range)
    flags = {name: int(bool(rx.search(full))) for name, rx in _COMPILED.items()}
    letters = [c for c in full if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
    flags["excessive_caps"] = int(caps > 0.30)
    flags["missing_company_profile"] = int(not str(company_profile or "").strip())
    flags["very_short"] = int(len(str(description or "").split()) < 15)
    return {name: flags[name] for name in FLAG_NAMES}
