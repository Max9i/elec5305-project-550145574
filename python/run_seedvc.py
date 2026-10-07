"""Seed-VC V2 in anonymization-only mode for every trial utterance (README §4.2, M4).

Runs in the "seedvc" conda env (python/requirements-seedvc.txt), from the repository root:

    python python/run_seedvc.py 0.7      # intelligibility CFG rate -> data/anon/seedvc_cfg0.7/<utt>.wav

Pretrained checkpoints only, no training. In anonymization-only mode Seed-VC zeroes the reference
speaker prompt and style vector (modules/v2/cfm.py, random_voice branch), so similarity_cfg_rate has
no effect and intelligibility_cfg_rate is the only guidance knob left. The API still needs a
reference file; 1 s of silence is passed so that no other speaker's audio enters the conditioning.
"""
import csv
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDVC = ROOT / "third_party/seed-vc"
SEED, STEPS, DEV = 5305, 30, torch.device("cuda")

sys.path.insert(0, str(SEEDVC))
os.chdir(SEEDVC)  # Seed-VC reads its config and caches checkpoints relative to its own folder
import modules.v2.vc_wrapper as vcw  # noqa: E402

vcw.AudioSegment.export = lambda self, *a, **k: io.BytesIO()  # skip per-chunk MP3 encoding (needs ffmpeg)


def load():
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    m = instantiate(DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml"))))
    m.load_checkpoints()
    return m.to(DEV).eval()


def convert(m, src, ref, cfg_rate):
    torch.manual_seed(SEED)  # same diffusion noise for every utterance and run
    for _, full in m.convert_voice_with_streaming(
            source_audio_path=str(src), target_audio_path=str(ref), diffusion_steps=STEPS, length_adjust=1.0,
            intelligebility_cfg_rate=cfg_rate, similarity_cfg_rate=0.7, convert_style=False,
            anonymization_only=True, device=DEV, dtype=torch.float16, stream_output=True):
        pass
    return full  # (sample rate, waveform) of the whole utterance, from the last chunk


def main(cfg_rate, *utts):
    cfg_rate = float(cfg_rate)
    cond = f"seedvc_cfg{cfg_rate}"
    out = ROOT / "data/anon" / cond
    out.mkdir(parents=True, exist_ok=True)
    ref = ROOT / "data/anon/silence_1s.wav"
    sf.write(ref, np.zeros(22050, dtype="float32"), 22050)
    utts = utts or [r["utt"] for r in csv.DictReader(open(ROOT / "data/lists/trials.csv"))]

    m = load()
    audio_s = out_s = proc_s = clipped = 0
    for u in utts:
        spk, chap, _ = u.split("-")
        src = ROOT / "data/raw/LibriSpeech/test-clean" / spk / chap / f"{u}.flac"
        t = time.time()
        sr, y = convert(m, src, ref, cfg_rate)
        torch.cuda.synchronize()
        proc_s += time.time() - t
        audio_s += sf.info(src).duration
        out_s += len(y) / sr
        clipped += int(np.abs(y).max() >= 0.999)  # BigVGAN clamps its output to +-1
        sf.write(out / f"{u}.wav", y, sr, subtype="FLOAT")
    print(f"{cond}: {audio_s:.0f} s in, {out_s:.0f} s out, {proc_s:.0f} s, {clipped} clipped utterances")

    path = ROOT / "results/timing_seedvc.csv"
    rows = [r for r in (csv.DictReader(open(path)) if path.exists() else []) if r["condition"] != cond]
    rows.append(dict(condition=cond, audio_s=round(audio_s, 2), out_s=round(out_s, 2), proc_s=round(proc_s, 2),
                     rtf=round(proc_s / audio_s, 4), clipped_utts=clipped))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[-1].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:])
