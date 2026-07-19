"""Generate data/team_flags.json mapping every rated national team to a flag
code for https://flagcdn.com. Uses pycountry with a manual override table for
names that don't match cleanly (England/Scotland/Wales, Korea, etc.).

Run:  python scripts/build_flags.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OVERRIDES = {
    "England": "gb-eng", "Scotland": "gb-sct", "Wales": "gb-wls",
    "Northern Ireland": "gb-nir", "Republic of Ireland": "ie", "Ireland": "ie",
    "South Korea": "kr", "Korea Republic": "kr", "North Korea": "kp",
    "Korea DPR": "kp", "United States": "us", "USA": "us",
    "Ivory Coast": "ci", "Côte d'Ivoire": "ci", "Cote d'Ivoire": "ci",
    "DR Congo": "cd", "Congo DR": "cd", "Congo": "cg", "Congo Republic": "cg",
    "Cape Verde": "cv", "Cape Verde Islands": "cv", "Cabo Verde": "cv",
    "Curaçao": "cw", "Curacao": "cw", "Chinese Taipei": "tw", "Taiwan": "tw",
    "China PR": "cn", "China": "cn", "Iran": "ir", "Russia": "ru",
    "Bolivia": "bo", "Venezuela": "ve", "Tanzania": "tz", "Syria": "sy",
    "Palestine": "ps", "Kosovo": "xk", "Hong Kong": "hk", "Macau": "mo",
    "Brunei": "bn", "Vietnam": "vn", "Laos": "la", "Moldova": "md",
    "North Macedonia": "mk", "Macedonia": "mk", "Czech Republic": "cz",
    "Czechia": "cz", "Turkey": "tr", "Türkiye": "tr", "UAE": "ae",
    "United Arab Emirates": "ae", "Eswatini": "sz", "Swaziland": "sz",
    "Cape Verde": "cv", "São Tomé and Príncipe": "st", "Sao Tome and Principe": "st",
    "St Kitts and Nevis": "kn", "Saint Kitts and Nevis": "kn",
    "St Lucia": "lc", "Saint Lucia": "lc",
    "St Vincent and the Grenadines": "vc", "Saint Vincent and the Grenadines": "vc",
    "Antigua and Barbuda": "ag", "Trinidad and Tobago": "tt",
    "Bosnia and Herzegovina": "ba", "Tahiti": "pf", "New Caledonia": "nc",
    "Guadeloupe": "gp", "Martinique": "mq", "French Guiana": "gf",
    "Bermuda": "bm", "Puerto Rico": "pr", "Guam": "gu", "Aruba": "aw",
    "Sint Maarten": "sx", "Bonaire": "bq", "Turks and Caicos Islands": "tc",
    "British Virgin Islands": "vg", "US Virgin Islands": "vi",
    "Cayman Islands": "ky", "Montserrat": "ms", "Anguilla": "ai",
    "American Samoa": "as", "Cook Islands": "ck", "Gibraltar": "gi",
    "Faroe Islands": "fo", "Kiribati": "ki", "Tuvalu": "tv", "Niue": "nu",
    "Zanzibar": "tz", "Réunion": "re", "Mayotte": "yt",
    "Bhutan": "bt", "Timor-Leste": "tl", "East Timor": "tl",
}


def resolve(name: str) -> str | None:
    if name in OVERRIDES:
        return OVERRIDES[name]
    try:
        import pycountry
    except ImportError:
        return None
    try:
        c = pycountry.countries.get(name=name)
        if c is None:
            c = pycountry.countries.get(common_name=name)
        if c is None:
            res = pycountry.countries.search_fuzzy(name)
            c = res[0] if res else None
        return c.alpha_2.lower() if c else None
    except (LookupError, Exception):
        return None


def main():
    ratings = json.loads((config.PROCESSED_DIR / "current_ratings.json").read_text())
    flags, missing = {}, []
    for team in ratings:
        code = resolve(team)
        if code:
            flags[team] = code
        else:
            missing.append(team)
    (config.ROOT / "data" / "team_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False))
    print(f"resolved {len(flags)} flags; {len(missing)} unresolved")
    if missing:
        print("unresolved (will show a neutral placeholder):", ", ".join(sorted(missing)[:40]))


if __name__ == "__main__":
    main()
