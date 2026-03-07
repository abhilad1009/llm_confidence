"""
shared/value_generator.py — Produce type-aware random alternative values.

Used in Stage 2 (localisation): we inject synthetic values into the anchor
and ask the model to extract them, verifying it reads from the right location.

Supported types detected automatically:
  • integer / float
  • currency  ($1,234.56)
  • date
  • phone number
  • email
  • generic string / name
"""

from __future__ import annotations

import re
import random
import string
from datetime import date, timedelta
from typing import Callable

# Optional Faker for richer name/address generation
try:
    from faker import Faker
    _faker = Faker()
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False


# ──────────────────────────────────────────────────────────────────────────────
# Type Detection
# ──────────────────────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
    r"\d{4}-\d{2}-\d{2}",
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}\b",
]

_PHONE_PATTERN = r"(\+?\d[\d\s\-().]{6,}\d)"
_EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"
_CURRENCY_PATTERN = r"^\$?[\d,]+(?:\.\d{2})?$"


def detect_value_type(value: str) -> str:
    v = value.strip()
    # Currency
    if re.match(_CURRENCY_PATTERN, v):
        try:
            float(v.lstrip("$").replace(",", ""))
            return "currency" if "$" in v or "," in v else "number"
        except ValueError:
            pass
    # Plain number
    try:
        float(v.replace(",", ""))
        return "number"
    except ValueError:
        pass
    # Date
    for pat in _DATE_PATTERNS:
        if re.search(pat, v, re.IGNORECASE):
            return "date"
    # Phone
    if re.fullmatch(_PHONE_PATTERN, v):
        return "phone"
    # Email
    if re.fullmatch(_EMAIL_PATTERN, v):
        return "email"
    return "string"


# ──────────────────────────────────────────────────────────────────────────────
# Generators per type
# ──────────────────────────────────────────────────────────────────────────────

def _gen_numbers(value: str, n: int) -> list[str]:
    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        base = float(cleaned)
    except ValueError:
        base = 100.0
    has_decimal = "." in cleaned
    results = []
    for _ in range(n):
        factor = random.uniform(0.3, 3.0)
        new_val = base * factor
        results.append(f"{new_val:.2f}" if has_decimal else str(int(new_val)))
    return results


def _gen_currencies(value: str, n: int) -> list[str]:
    cleaned = re.sub(r"[^\d.]", "", value)
    try:
        base = float(cleaned)
    except ValueError:
        base = 100.0
    results = []
    prefix = "$" if "$" in value else ""
    for _ in range(n):
        factor = random.uniform(0.2, 5.0)
        results.append(f"{prefix}{base * factor:,.2f}")
    return results


def _gen_dates(value: str, n: int) -> list[str]:
    if re.match(r"\d{4}-\d{2}-\d{2}", value):
        fmt = "%Y-%m-%d"
    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", value):
        fmt = "%m/%d/%Y"
    elif re.match(r"\d{1,2}-\d{1,2}-\d{4}", value):
        fmt = "%m-%d-%Y"
    else:
        fmt = "%B %d, %Y"
    base = date.today()
    results = []
    for _ in range(n):
        delta = random.randint(-730, 730)
        new_date = base + timedelta(days=delta)
        try:
            results.append(new_date.strftime(fmt))
        except Exception:
            results.append(new_date.isoformat())
    return results


def _gen_phones(value: str, n: int) -> list[str]:
    results = []
    for _ in range(n):
        if _HAS_FAKER:
            results.append(_faker.phone_number())
        else:
            digits = "".join(str(random.randint(0, 9)) for _ in range(10))
            results.append(f"({digits[:3]}) {digits[3:6]}-{digits[6:]}")
    return results


def _gen_emails(value: str, n: int) -> list[str]:
    if _HAS_FAKER:
        return [_faker.email() for _ in range(n)]
    results = []
    for _ in range(n):
        user = "".join(random.choices(string.ascii_lowercase, k=random.randint(5, 10)))
        domain = "".join(random.choices(string.ascii_lowercase, k=random.randint(4, 8)))
        tld = random.choice(["com", "net", "org", "io"])
        results.append(f"{user}@{domain}.{tld}")
    return results


_WORD_LIST = [
    "apple", "bridge", "castle", "dragon", "engine", "forest", "garden",
    "harbor", "island", "jungle", "knight", "lantern", "meadow", "needle",
    "ocean", "pillar", "quarry", "rabbit", "silver", "thunder", "umbrella",
    "valley", "window", "yellow", "zenith", "anchor", "barrel", "candle",
    "dolphin", "ember", "falcon", "glacier", "hammer", "iron", "jewel",
]

def _gen_strings(value: str, n: int) -> list[str]:
    words = value.split()
    num_words = max(1, len(words))
    results = []
    for _ in range(n):
        if _HAS_FAKER and num_words <= 3:
            results.append(_faker.name() if num_words >= 2 else _faker.first_name())
        else:
            chosen = random.choices(_WORD_LIST, k=num_words)
            out = []
            for orig_w, new_w in zip(words, chosen):
                out.append(new_w.capitalize() if orig_w and orig_w[0].isupper() else new_w)
            results.append(" ".join(out))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_GENERATORS: dict[str, Callable] = {
    "number":   _gen_numbers,
    "currency": _gen_currencies,
    "date":     _gen_dates,
    "phone":    _gen_phones,
    "email":    _gen_emails,
    "string":   _gen_strings,
}


class ValueGenerator:
    """Generates type-aware random alternative values for perturbation testing."""

    def generate(self, value: str, n: int = 10, exclude: list[str] | None = None) -> list[str]:
        vtype = detect_value_type(value)
        gen_fn = _GENERATORS.get(vtype, _gen_strings)
        candidates = gen_fn(value, n * 2)

        exclude_set = set((exclude or []) + [value])
        results = [c for c in candidates if c not in exclude_set and c.strip()]
        seen: set[str] = set()
        unique = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique[:n]

    def detected_type(self, value: str) -> str:
        return detect_value_type(value)
