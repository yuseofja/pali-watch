"""
Keyword-score an HDX catalog CSV to find datasets likely to match a topic.

Typical use (in Spyder or a notebook, from the project root):

    from search_hdx_catalogue import search_catalog
    cands = search_catalog("data/processed/hdx_pse_catalog_raw.csv", preset="displacement")
    cands = search_catalog(path, preset="displacement", min_score=2, top_n=20)

Custom topic:

    signals = {
        "wash":   (r"\\bwash\\b|water|sanitation|hygiene", 3),
        "gov":    (r"governorates?", 1),
    }
    cands = search_catalog(path, signals=signals, min_score=3)

A "signal" is name -> (regex, weight). Each signal counts once per dataset,
no matter how many times it matches. A dataset's score is the sum of the weights
of the signals it matches. Matching is case-insensitive.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

Signals = Mapping[str, tuple[str, float]]

# --------------------------------------------------------------------------
# Presets. Strong, topic-specific phrases get high weights; generic supporting
# words get low weights. Edit freely or add your own.
# --------------------------------------------------------------------------
PRESETS: dict[str, dict[str, tuple[str, float]]] = {
    "3w": {
        "3w":         (r"\b[345]w\b", 5),
        "who_what":   (r"who(\s+is)?\s+doing\s+what(\s+and)?\s+where", 5),
        "presence":   (r"humanitarian (presence|response)|service points?", 3),
        "partners":   (r"\b(partners?|organi[sz]ations?|implementing)\b", 1),
        "clusters":   (r"\b(clusters?|sectors?)\b", 1),
        "activities": (r"\b(activit(y|ies)|programmes?|projects?)\b", 1),
        "reach":      (r"\b(beneficiar\w*|people reached|reached)\b", 1),
        "geo":        (r"\b(neighbou?rhoods?|governorates?|admin ?[1-4])\b", 1),
    },
    "displacement": {
        "idp":        (r"\b(idps?|internally[- ]displaced)\b", 5),
        "displace":   (r"\bdisplace\w*", 4),
        "demolition": (r"\bdemolition\w*|\bevictions?\b", 3),
        "movement":   (r"population movements?|flow monitoring|\bdtm\b", 3),
        "shelter":    (r"\b(shelters?|sites?|camps?)\b", 1),
        "geo":        (r"\b(governorates?|neighbou?rhoods?|admin ?[1-4])\b", 1),
    },
    "food_security": {
        "ipc":        (r"\bipc\b|integrated food security", 5),
        "food_insec": (r"food insecur\w*|food security|\bhunger\b|famine", 4),
        "consumption": (r"food consumption score|\bfcs\b|\bhdds\b|\bmvam\b", 3),
        "nutrition":  (r"malnutrition|nutrition|\bmuac\b|\bgam\b", 2),
        "prices":     (r"food prices?|market monitor", 1),
        "geo":        (r"\b(governorates?|admin ?[1-4])\b", 1),
    },
    "health": {
        "herams":     (r"\bherams\b", 5),
        "facilities": (r"health (facilit\w+|care|system|services?)|hospitals?|clinics?", 3),
        "attacks":    (r"attacks? on health|\bssa\b", 3),
        "functional": (r"functional\w*|operational", 1),
        "geo":        (r"\b(governorates?|admin ?[1-4])\b", 1),
    },
    "damage": {
        "unosat":     (r"\bunosat\b|damage assessment", 5),
        "damage":     (r"\bdamage[sd]?\b|destroyed|destruction", 3),
        "buildings":  (r"\b(buildings?|housing units?|infrastructure)\b", 2),
        "satellite":  (r"satellite|remote sensing", 1),
    },
    "access": {
        "access":     (r"humanitarian access|movement and access", 5),
        "obstacles":  (r"\b(checkpoints?|roadblocks?|obstacles?|barriers?)\b", 4),
        "crossings":  (r"\bcrossings?\b|closures?", 2),
        "restrict":   (r"restrictions?", 1),
    },
    "boundaries": {
        "codab":      (r"\bcod[- ]?ab\b|administrative boundar\w*", 5),
        "admin":      (r"\badmin(istrative)? ?(level)?\s?[0-4]\b|\bp-?codes?\b", 3),
        "shape":      (r"shapefile|geojson|\bgdb\b", 1),
    },
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_catalog(path: str | Path, encodings: Iterable[str] = ("utf-8", "cp1252")) -> pd.DataFrame:
    """Read a catalog CSV, trying each encoding in turn (Windows exports are often cp1252)."""
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:
            last_err = err
    raise last_err  # type: ignore[misc]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_catalog(
    df: pd.DataFrame,
    signals: Signals,
    text_cols: Iterable[str] = ("title", "notes"),
) -> pd.DataFrame:
    """
    Return a copy of df with two added columns:
      score    sum of weights of signals that matched
      matched  comma-separated names of the matching signals
    """
    text_cols = [c for c in text_cols if c in df.columns]
    if not text_cols:
        raise KeyError(f"None of the text columns are in the catalog. Columns: {list(df.columns)}")

    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

    # One boolean column per signal, then derive score and matched-names from it.
    # (re.search per row, not str.contains, so patterns with groups don't trigger pandas warnings)
    compiled = {name: re.compile(pattern, re.IGNORECASE) for name, (pattern, _) in signals.items()}
    hit_matrix = pd.DataFrame(
        {name: text.map(lambda t, rx=rx: rx.search(t) is not None) for name, rx in compiled.items()},
        index=df.index,
    )
    weights = pd.Series({name: w for name, (_, w) in signals.items()})

    out = df.copy()
    out["score"] = hit_matrix.mul(weights, axis=1).sum(axis=1)
    out["matched"] = hit_matrix.apply(lambda r: ", ".join(r.index[r.to_numpy()]), axis=1)
    return out


def find_candidates(
    df: pd.DataFrame,
    signals: Signals,
    min_score: float = 3,
    text_cols: Iterable[str] = ("title", "notes"),
    exclude: str | None = None,
    top_n: int | None = None,
    date_col: str = "last_modified",
) -> pd.DataFrame:
    """Score, filter by min_score, optionally drop rows matching `exclude`, and sort."""
    scored = score_catalog(df, signals, text_cols)
    if exclude:
        cols = [c for c in text_cols if c in scored.columns]
        text = scored[cols].fillna("").astype(str).agg(" ".join, axis=1)
        ex = re.compile(exclude, re.IGNORECASE)
        scored = scored[~text.map(lambda t: ex.search(t) is not None)]

    cands = scored[scored["score"] >= min_score].copy()

    # Sort by score, then most recently modified first (parse dates only for sorting).
    if date_col in cands.columns:
        cands["_sort_date"] = pd.to_datetime(cands[date_col], errors="coerce", utc=True)
        cands = cands.sort_values(["score", "_sort_date"], ascending=False).drop(columns="_sort_date")
    else:
        cands = cands.sort_values("score", ascending=False)

    return cands.head(top_n) if top_n else cands


# --------------------------------------------------------------------------
# One-call convenience wrapper
# --------------------------------------------------------------------------
def search_catalog(
    path: str | Path,
    preset: str | None = None,
    signals: Signals | None = None,
    min_score: float = 3,
    text_cols: Iterable[str] = ("title", "notes"),
    exclude: str | None = None,
    top_n: int | None = None,
    show_cols: Iterable[str] = ("title", "score", "matched", "last_modified"),
    out_path: str | Path | None = "auto",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a catalog CSV, score it, and return the ranked candidates.

    preset    name from PRESETS (e.g. "3w", "displacement"). Ignored if `signals` is given.
    signals   your own {name: (regex, weight)}.
    out_path  "auto" -> data/processed/<preset or 'custom'>_candidates.csv next to the input;
              a path -> save there; None -> don't save.
    """
    if signals is None:
        if preset is None:
            raise ValueError(f"Give a preset ({', '.join(PRESETS)}) or custom signals.")
        if preset not in PRESETS:
            raise KeyError(f"Unknown preset '{preset}'. Available: {', '.join(PRESETS)}")
        signals = PRESETS[preset]
    label = preset if (preset and signals is PRESETS.get(preset)) else "custom"

    df = load_catalog(path)
    cands = find_candidates(df, signals, min_score, text_cols, exclude, top_n)

    if verbose:
        cols = [c for c in show_cols if c in cands.columns]
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(f"{len(cands)} of {len(df)} datasets scored >= {min_score} ({label})")
            print(cands[cols].to_string(index=False))

    if out_path == "auto":
        out_path = Path(path).parent / f"{label}_candidates.csv"
    if out_path is not None:
        cands.to_csv(out_path, index=False)
        if verbose:
            print(f"\nSaved to {out_path}")
    return cands


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Score an HDX catalog CSV for a topic.")
    p.add_argument("catalog", help="path to catalog CSV")
    p.add_argument("--preset", default="3w", choices=sorted(PRESETS))
    p.add_argument("--min-score", type=float, default=3)
    p.add_argument("--top-n", type=int, default=None)
    a = p.parse_args()
    search_catalog(a.catalog, preset=a.preset, min_score=a.min_score, top_n=a.top_n)