"""Acoustic change between original and anonymised trial utterances (README §6, M3).

    python python/acoustics.py mcadams_0.8 data/anon/mcadams_0.8

Frames are matched by position: the anonymised frame grid is linearly stretched onto the
original's, which is exact for McAdams and tolerates small duration changes. Per utterance:
  f0_shift_st  median F0 change (semitones) over frames voiced in both (pYIN, 50-500 Hz)
  f0_corr      correlation of log-F0 over the same frames (is the intonation shape kept?)
  voiced_kept  share of the original's voiced frames that are still voiced (harmonicity kept?)
  env_lsd_db   RMS dB difference between the gain-normalised order-20 LPC envelopes 1/|A|,
               averaged over frames within 40 dB of the loudest original frame
Writes results/acoustics/<cond>.csv and one row of results/acoustics.csv (medians over utterances
for the F0 measures, which pYIN octave errors would skew, means for the others).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIBRI = ROOT / "data/raw/LibriSpeech/test-clean"
RES = ROOT / "results"
SR, HOP = 16000, 160


def match(n_from, n_to):
    """Index into a sequence of n_from frames for each of n_to frames, by relative position."""
    return np.round(np.linspace(0, n_from - 1, n_to)).astype(int)


def f0(x):
    f, voiced, _ = librosa.pyin(x, fmin=50, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
    return np.where(voiced, f, np.nan)


def envelopes(x):
    """dB envelope 1/|A(e^jw)| of every 20 ms frame (Hann, 10 ms hop), plus each frame's energy in dB."""
    frames = librosa.util.frame(x, frame_length=2 * HOP, hop_length=HOP, axis=0) * np.hanning(2 * HOP)
    frames = frames + 1e-6 * np.random.default_rng(0).standard_normal(frames.shape)  # LPC needs non-silent frames
    a = librosa.lpc(frames, order=20, axis=-1)
    return -20 * np.log10(np.abs(np.fft.rfft(a, 512, axis=-1))), 10 * np.log10(np.sum(frames ** 2, axis=-1))


def compare(x, y):
    fx, fy = f0(x), f0(y)
    fy = fy[match(len(fy), len(fx))]
    both = ~np.isnan(fx) & ~np.isnan(fy)
    kept = both.sum() / max((~np.isnan(fx)).sum(), 1)
    st = 12 * np.log2(fy[both] / fx[both])
    shift = np.median(st) if both.sum() >= 10 else np.nan
    corr = np.corrcoef(np.log(fx[both]), np.log(fy[both]))[0, 1] if both.sum() >= 10 else np.nan

    (ex, energy), (ey, _) = envelopes(x), envelopes(y)
    ey = ey[match(len(ey), len(ex))]
    speech = energy > energy.max() - 40
    lsd = np.mean(np.sqrt(np.mean((ex[speech] - ey[speech]) ** 2, axis=-1)))
    return shift, corr, kept, lsd


def one(utt, wav_dir):
    spk, chap, _ = utt.split("-")
    x = librosa.load(LIBRI / spk / chap / f"{utt}.flac", sr=SR)[0]
    y = librosa.load(Path(wav_dir) / f"{utt}.wav", sr=SR)[0]
    return compare(x, y)


def main(cond, wav_dir):
    utts = pd.read_csv(ROOT / "data/lists/trials.csv", dtype=str).utt
    with ProcessPoolExecutor(max_workers=8) as ex:  # 16 ran out of commit memory next to a GPU job
        rows = list(ex.map(partial(one, wav_dir=wav_dir), utts, chunksize=4))
    df = pd.DataFrame(rows, columns=["f0_shift_st", "f0_corr", "voiced_kept", "env_lsd_db"]).assign(utt=utts.values)
    (RES / "acoustics").mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(RES / "acoustics" / f"{cond}.csv", index=False)

    agg = {"f0_shift_st": "median", "f0_corr": "median", "voiced_kept": "mean", "env_lsd_db": "mean"}
    summary = {"condition": cond, **df.agg(agg).round(3).to_dict()}
    path = RES / "acoustics.csv"
    rows = [r for r in (pd.read_csv(path).to_dict("records") if path.exists() else []) if r["condition"] != cond]
    pd.DataFrame(rows + [summary]).to_csv(path, index=False)
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:])
