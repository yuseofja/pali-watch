# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:04:23 2026

@author: youse
"""

import re
import pandas as pd

df = pd.read_csv("data/processed/hdx_pse_catalog_raw.csv", encoding="cp1252")

# label: (regex, weight). Strong 3W phrases weigh more than generic coverage terms.
SIGNALS = {
    "3w":        (r"\b[345]w\b", 5),
    "who_what":  (r"who(\s+is)?\s+doing\s+what(\s+and)?\s+where", 5),
    "presence":  (r"humanitarian (presence|response)|service points?", 3),
    "partners":  (r"\b(partners?|organi[sz]ations?|implementing)\b", 1),
    "clusters":  (r"\b(clusters?|sectors?)\b", 1),
    "activities": (r"\b(activit(y|ies)|programmes?|projects?)\b", 1),
    "reach":     (r"\b(beneficiar\w*|people reached|reached)\b", 1),
    "geo":       (r"\b(neighbou?rhoods?|governorates?|admin ?[1-4])\b", 1),
}

def score(row):
    text = f"{row['title']} {row['notes']}".lower()
    hits = [k for k, (pat, _) in SIGNALS.items() if re.search(pat, text)]
    return sum(SIGNALS[k][1] for k in hits), ", ".join(hits)

df[["score", "matched"]] = df.apply(lambda r: pd.Series(score(r)), axis=1)

cands = (df[df["score"] >= 3]
         .sort_values(["score", "last_modified"], ascending=False))

pd.set_option("display.max_colwidth", 60, "display.width", 200)
print(cands[["title", "score", "matched", "last_modified"]])
cands.to_csv("data/processed/3w_candidates.csv", index=False)