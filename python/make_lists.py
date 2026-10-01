"""Build the fixed enrollment/trial lists from LibriSpeech test-clean.

Run once and commit data/lists/. Every anonymiser and attacker reuses these lists.
"""
import random
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LIBRI = ROOT / "data/raw/LibriSpeech"
OUT = ROOT / "data/lists"
SEED, N_ENROLL, N_TRIAL = 5305, 5, 10
MIN_DUR, MAX_DUR = 4.0, 15.0  # s: short clips give unstable embeddings, long ones slow Seed-VC

# SPEAKERS.TXT columns: ID | SEX | SUBSET | MINUTES | NAME
sex = {}
for line in (LIBRI / "SPEAKERS.TXT").read_text(encoding="utf-8").splitlines():
    f = [x.strip() for x in line.split("|")]
    if not line.startswith(";") and len(f) > 2 and f[2] == "test-clean":
        sex[f[0]] = f[1]

rows = []
for trans in sorted((LIBRI / "test-clean").glob("*/*/*.trans.txt")):
    for line in trans.read_text(encoding="utf-8").splitlines():
        utt, text = line.split(" ", 1)
        spk = utt.split("-")[0]
        dur = sf.info(trans.parent / f"{utt}.flac").duration
        rows.append((utt, spk, sex[spk], round(dur, 2), text))
utts = pd.DataFrame(rows, columns=["utt", "spk", "sex", "dur", "text"])
ok = utts[utts.dur.between(MIN_DUR, MAX_DUR)]

rng = random.Random(SEED)
enroll, trial = [], []
for spk, g in ok.groupby("spk"):
    ids = sorted(g.utt)
    assert len(ids) >= N_ENROLL + N_TRIAL, f"speaker {spk} has only {len(ids)} usable utterances"
    rng.shuffle(ids)
    enroll += ids[:N_ENROLL]
    trial += ids[N_ENROLL:N_ENROLL + N_TRIAL]
enroll = ok[ok.utt.isin(enroll)].drop(columns="text")
trial = ok[ok.utt.isin(trial)]

# Every trial is scored against every same-sex speaker model: own speaker = target, others = impostor
spk_sex = ok.drop_duplicates("spk").set_index("spk").sex
pairs = pd.DataFrame(
    [(u, m, int(s == m)) for u, s, x in trial[["utt", "spk", "sex"]].itertuples(index=False)
     for m in spk_sex.index[spk_sex == x]],
    columns=["trial", "model_spk", "target"])

OUT.mkdir(parents=True, exist_ok=True)
spk_sex.reset_index().to_csv(OUT / "speakers.csv", index=False)
enroll.to_csv(OUT / "enroll.csv", index=False)
trial.to_csv(OUT / "trials.csv", index=False)
pairs.to_csv(OUT / "pairs.csv", index=False)
print(f"{len(spk_sex)} speakers ({(spk_sex == 'F').sum()} F), {len(enroll)} enroll, {len(trial)} trials, "
      f"{pairs.target.sum()} target / {(1 - pairs.target).sum()} impostor pairs")
