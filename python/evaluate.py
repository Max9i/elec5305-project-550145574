"""Privacy (ECAPA / WavLM-SV EER) and utility (ASR WER) of one condition, on the fixed lists.

    python python/evaluate.py original                             # LibriSpeech trial audio
    python python/evaluate.py mcadams_0.8 data/anon/mcadams_0.8    # folder of <utt>.wav, any rate

Enrollment is always original speech (ignorant attacker). Writes results/scores/<cond>.csv,
results/asr/<cond>.csv, results/figures/scores_<cond>.png and one row of results/summary.csv.
"""
import sys
from pathlib import Path

import jiwer
import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LIBRI = ROOT / "data/raw/LibriSpeech/test-clean"
LISTS = ROOT / "data/lists"
RES = ROOT / "results"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SR = 16000
ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
WAVLM = "microsoft/wavlm-base-plus-sv"
ASR = "facebook/wav2vec2-large-960h-lv60-self"
NAMES = {"ecapa": "ECAPA-TDNN", "wavlm": "WavLM-SV"}


def eer(scores, target):
    """Equal error rate in %: sweep the threshold over the sorted scores, take FAR ~= FRR."""
    t = np.asarray(target)[np.argsort(scores)]
    frr = np.r_[0, np.cumsum(t)] / t.sum()                # targets rejected below the threshold
    far = 1 - np.r_[0, np.cumsum(1 - t)] / (1 - t).sum()  # impostors accepted above it
    i = np.argmin(np.abs(frr - far))
    return 100 * (frr[i] + far[i]) / 2


def libri(utt):
    spk, chap, _ = utt.split("-")
    return LIBRI / spk / chap / f"{utt}.flac"


def load(path):
    """Mono float32 at 16 kHz, scaled to a fixed RMS: WavLM-SV's feature extractor does not normalise
    its input (an utterance vs. itself at -26 dB: cosine 0.985), and anonymisers change the level."""
    x = librosa.load(path, sr=SR)[0]
    return 0.05 * x / (np.sqrt(np.mean(x ** 2)) + 1e-9)


def ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    m = EncoderClassifier.from_hparams(ECAPA, savedir=ROOT / "data/models/ecapa", run_opts={"device": DEV},
                                       local_strategy=LocalStrategy.COPY)  # Windows symlinks need admin rights
    return lambda x: m.encode_batch(torch.from_numpy(x)[None].to(DEV)).squeeze()


def wavlm():
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained(WAVLM)
    m = WavLMForXVector.from_pretrained(WAVLM).to(DEV).eval()
    return lambda x: m(**fe(x, sampling_rate=SR, return_tensors="pt").to(DEV)).embeddings.squeeze()


def asr():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    p = Wav2Vec2Processor.from_pretrained(ASR)
    m = Wav2Vec2ForCTC.from_pretrained(ASR).to(DEV).eval()
    return lambda x: p.decode(m(p(x, sampling_rate=SR, return_tensors="pt").input_values.to(DEV)).logits[0].argmax(-1))


def plot(pairs, summary, cond):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                         "xtick.color": "#898781", "axes.labelcolor": "#52514e", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), facecolor="#fcfcfb")
    for ax, name in zip(axes, ("ecapa", "wavlm")):
        bins = np.linspace(pairs[name].min(), pairs[name].max(), 60)
        for t, label, color in [(1, "genuine", "#2a78d6"), (0, "impostor", "#eb6834")]:
            ax.hist(pairs[name][pairs.target == t], bins, density=True, histtype="step", lw=2, color=color, label=label)
        ax.set(facecolor="#fcfcfb", yticks=[], xlabel="cosine score",
               title=f"{cond} · {NAMES[name]} · EER {summary[name + '_eer']:.2f}%")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "figures" / f"scores_{cond}.png", dpi=150)


@torch.inference_mode()
def main(cond, wav_dir=None):
    enroll = pd.read_csv(LISTS / "enroll.csv", dtype=str)
    trials = pd.read_csv(LISTS / "trials.csv", dtype=str)
    pairs = pd.read_csv(LISTS / "pairs.csv", dtype={"trial": str, "model_spk": str})
    enroll_wav = {u: load(libri(u)) for u in enroll.utt}
    trial_wav = {u: load(Path(wav_dir) / f"{u}.wav" if wav_dir else libri(u)) for u in trials.utt}

    summary = {"condition": cond}
    for name, make in [("ecapa", ecapa), ("wavlm", wavlm)]:
        embed = make()
        unit = lambda x: F.normalize(embed(x).float(), dim=-1)
        # speaker model = mean of the unit-length enrollment embeddings; score = cosine similarity
        models = {s: F.normalize(torch.stack([unit(enroll_wav[u]) for u in g.utt]).mean(0), dim=-1)
                  for s, g in enroll.groupby("spk")}
        emb = {u: unit(x) for u, x in trial_wav.items()}
        pairs[name] = [float(emb[u] @ models[s]) for u, s in zip(pairs.trial, pairs.model_spk)]
        summary[f"{name}_eer"] = round(float(eer(pairs[name], pairs.target)), 2)

    rec = asr()
    hyp = pd.DataFrame({"utt": trials.utt, "ref": trials.text, "hyp": [rec(trial_wav[u]) for u in trials.utt]})
    hyp["wer"] = [round(jiwer.wer(r, h), 4) for r, h in zip(hyp.ref, hyp.hyp)]
    summary["wer"] = round(100 * jiwer.wer(list(hyp.ref), list(hyp.hyp)), 2)

    for sub in ("scores", "asr", "figures"):
        (RES / sub).mkdir(parents=True, exist_ok=True)
    pairs.round(4).to_csv(RES / "scores" / f"{cond}.csv", index=False)
    hyp.to_csv(RES / "asr" / f"{cond}.csv", index=False)
    path = RES / "summary.csv"
    rows = [r for r in (pd.read_csv(path).to_dict("records") if path.exists() else []) if r["condition"] != cond]
    pd.DataFrame(rows + [summary]).to_csv(path, index=False)
    plot(pairs, summary, cond)
    print(summary)


if __name__ == "__main__":
    main(*sys.argv[1:])
