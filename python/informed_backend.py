"""M7: lazy-informed attacker plus a backend trained on its own anonymised speech. This approximates a
semi-informed attacker without retraining the neural models.

    python python/informed_backend.py original                             # reference: original speech
    python python/informed_backend.py mcadams_0.8 data/anon/mcadams_0.8    # anonymised backend data,
                                                                           # enrollment and trials

The attacker anonymises its own data with the same system and settings: data/lists/backend.csv,
20 utterances of each of the 40 dev-clean speakers (disjoint from the test speakers). It trains an
LDA (scikit-learn, shrinkage, 39 dimensions) on the frozen unit-length embeddings with speaker labels.
Enrollment and trial embeddings are projected, length-normalised and scored by cosine. Writes
results/scores/<cond>_backend.csv and one row of results/summary_backend.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from evaluate import LISTS, RES, ROOT, ecapa, eer, load, wavlm


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


@torch.inference_mode()
def main(cond, wav_dir=None):
    backend = pd.read_csv(LISTS / "backend.csv", dtype=str)
    enroll = pd.read_csv(LISTS / "enroll.csv", dtype=str)
    pairs = pd.read_csv(LISTS / "pairs.csv", dtype={"trial": str, "model_spk": str})
    trials = list(pairs.trial.drop_duplicates())

    def path(u, subset="test-clean"):
        spk, chap, _ = u.split("-")
        return Path(wav_dir) / f"{u}.wav" if wav_dir else ROOT / "data/raw/LibriSpeech" / subset / spk / chap / f"{u}.flac"

    wav = {u: load(path(u, s)) for u, s in zip(backend.utt, backend.subset)}
    wav |= {u: load(path(u)) for u in list(enroll.utt) + trials}

    summary = {"condition": f"{cond}_backend"}
    for name, make in [("ecapa", ecapa), ("wavlm", wavlm)]:
        embed = make()
        E = {u: F.normalize(embed(x).float(), dim=-1).cpu().numpy() for u, x in wav.items()}
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(np.stack([E[u] for u in backend.utt]), backend.spk)
        project = lambda us: unit(lda.transform(np.stack([E[u] for u in us])))
        models = {s: unit(project(g.utt).mean(0)) for s, g in enroll.groupby("spk")}
        emb = dict(zip(trials, project(trials)))
        pairs[name] = [float(emb[u] @ models[s]) for u, s in zip(pairs.trial, pairs.model_spk)]
        summary[f"{name}_eer"] = round(float(eer(pairs[name], pairs.target)), 2)

    pairs.round(4).to_csv(RES / "scores" / f"{cond}_backend.csv", index=False)
    path_csv = RES / "summary_backend.csv"
    rows = [r for r in (pd.read_csv(path_csv).to_dict("records") if path_csv.exists() else [])
            if r["condition"] != summary["condition"]]
    pd.DataFrame(rows + [summary]).to_csv(path_csv, index=False)
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:])
