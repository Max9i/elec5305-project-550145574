"""Build the fixed lists: enrollment/trials from LibriSpeech test-clean and, if dev-clean is present,
the attacker's own data for the informed backend (M7) from dev-clean, whose speakers are disjoint.

Run once and commit data/lists/. Every anonymiser and attacker reuses these lists.
"""
import random
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LIBRI = ROOT / "data/raw/LibriSpeech"
OUT = ROOT / "data/lists"
SEED, N_ENROLL, N_TRIAL, N_BACKEND = 5305, 5, 10, 20
MIN_DUR, MAX_DUR = 4.0, 15.0  # s: short clips give unstable embeddings, long ones slow Seed-VC

# SPEAKERS.TXT columns: ID | SEX | SUBSET | MINUTES | NAME
sex = {}
for line in (LIBRI / "SPEAKERS.TXT").read_text(encoding="utf-8").splitlines():
    f = [x.strip() for x in line.split("|")]
    if not line.startswith(";") and len(f) > 2:
        sex[f[0]] = f[1]


def read_subset(subset):
    """Utterances of 4-15 s in one LibriSpeech subset, with speaker, sex, duration and transcript."""
    rows = []
    for trans in sorted((LIBRI / subset).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, text = line.split(" ", 1)
            spk = utt.split("-")[0]
            dur = sf.info(trans.parent / f"{utt}.flac").duration
            rows.append((utt, spk, sex[spk], round(dur, 2), text))
    utts = pd.DataFrame(rows, columns=["utt", "spk", "sex", "dur", "text"])
    return utts[utts.dur.between(MIN_DUR, MAX_DUR)]


ok = read_subset("test-clean")
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

if (LIBRI / "dev-clean").exists():
    dev = read_subset("dev-clean")
    rng = random.Random(SEED + 1)
    pick = []
    for spk, g in dev.groupby("spk"):
        ids = sorted(g.utt)
        rng.shuffle(ids)
        pick += ids[:N_BACKEND]
    backend = dev[dev.utt.isin(pick)].drop(columns="text").assign(subset="dev-clean")
    backend.to_csv(OUT / "backend.csv", index=False)
    print(f"backend: {backend.spk.nunique()} dev-clean speakers ({(backend.drop_duplicates('spk').sex == 'F').sum()} F), "
          f"{len(backend)} utterances, {backend.dur.sum() / 3600:.2f} h")
