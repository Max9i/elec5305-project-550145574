# Python

Evaluation (attackers, ASR, acoustic change) and the Seed-VC runs.

## Environment

One conda env, `seedvc` (Python 3.10). The RTX 50-series GPU needs a CUDA 12.8 build of PyTorch:

```bash
git clone https://github.com/Plachtaa/seed-vc third_party/seed-vc   # then: git checkout 51383ef
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r python/requirements.txt
```

## Scripts

| Script | Purpose |
|---|---|
| `make_lists.py` | builds `data/lists/` once (see `data/README.md`); `backend.csv` only if dev-clean is present |
| `run_seedvc.py <cfg> [trials\|enroll\|backend]` | Seed-VC V2 anonymization-only for every utterance of the list → `data/anon/seedvc_cfg<cfg>/`; for trials also `results/timing_seedvc.csv` |
| `evaluate.py <condition> [wav_dir] [lazy]` | ECAPA EER, WavLM-SV EER and ASR WER for one condition; `lazy`: lazy-informed attacker (anonymized enrollment from `wav_dir`, attackers only) → `results/summary_lazy.csv`, `results/scores/<condition>_lazy.csv` |
| `acoustics.py <condition> <wav_dir>` | F0 shift, F0 correlation, voicing kept and LPC-envelope distance vs. the original |
| `informed_backend.py <condition> [wav_dir]` | M7: lazy-informed attacker plus an LDA backend trained on the attacker's own anonymized dev-clean speech → `results/summary_backend.csv`, `results/scores/<condition>_backend.csv` |
| `plot_m5.py` | M5: Figures 1 and 2, speaker-bootstrap EER intervals, attacker gap and ranking comparison (printed) |
| `analyze_m6.py` | M6 statistics (printed): identified-trial categories, speaker-level agreement, feature correlations, session-cue check |
| `plot_m6.py` | M6: Figures 3, 4 and 5 |
| `test_eer.py`, `test_acoustics.py` | self-checks of the EER function and the acoustic measures |

`wav_dir` holds one `<utt>.wav` per trial utterance, at any sample rate (everything is resampled to
16 kHz). Without `wav_dir`, `evaluate.py` uses the original LibriSpeech audio.

Outputs:

- `results/summary.csv`: EERs and WER, one row per condition
- `results/acoustics.csv`: mean acoustic change, one row per condition (`results/acoustics/<condition>.csv` per utterance)
- `results/scores/<condition>.csv`: cosine score of every pair for both attackers
- `results/asr/<condition>.csv`: ASR hypothesis and per-utterance WER
- `results/figures/scores_<condition>.png`: genuine vs. impostor score histograms
- `results/timing_mcadams.csv`, `results/timing_seedvc.csv`: processing time per condition
- `results/eer_ci.csv`: EER with 95 % speaker-bootstrap interval per condition and attacker (`<condition>_lazy` rows: M7)
- `results/summary_lazy.csv`, `results/summary_backend.csv`: EERs of the lazy-informed attacker and of lazy-informed + adapted backend (M7)
- `results/figures/fig1_privacy_utility.png`, `fig2_attacker_disagreement.png`: M5 figures; `fig6_lazy_informed.png`: M7
- `results/figures/fig3_acoustic_change_vs_privacy.png`, `fig4_example_analysis.png`, `fig5_speaker_risk.png`: M6 figures
- `results/acoustics/original_f0.csv`: median F0 of every original trial utterance (cache for M6)

## Evaluation models (pretrained, frozen)

| Role | Model | Revision |
|---|---|---|
| Attacker A | `speechbrain/spkrec-ecapa-voxceleb` | `0f99f2d` |
| Attacker B | `microsoft/wavlm-base-plus-sv` | `feb593a` |
| ASR | `facebook/wav2vec2-large-960h-lv60-self`, greedy CTC decoding. Its output is already in LibriSpeech's upper-case transcript format, so no text normalization is needed | `54074b1` |

Speaker model = mean of the unit-length enrollment embeddings. Score = cosine similarity between
the trial embedding and the speaker model. Every waveform is scaled to the same RMS level before
scoring, because WavLM-SV's feature extractor does not normalize its input.

## Seed-VC settings

- Code: `Plachtaa/seed-vc` @ `51383ef`, config `configs/v2/vc_wrapper.yaml` (22.05 kHz output).
- Checkpoints (V2 defaults from `load_checkpoints()`): `Plachta/Seed-VC` @ `257283f`,
  `Plachta/ASTRAL-quantization` @ `4a2e967`, `funasr/campplus` @ `e4b6ede`,
  `nvidia/bigvgan_v2_22khz_80band_256x` @ `633ff70`, `facebook/hubert-large-ll60k` @ `ff022d0`,
  `openai/whisper-small` @ `973afd2`.
- `anonymization_only=True`, `convert_style=False`, `diffusion_steps=30`, `length_adjust=1.0`,
  intelligibility CFG 0.0 / 0.7 / 1.0 (`similarity_cfg_rate` has no effect in this mode),
  reference = 1 s of silence, `torch.manual_seed(5305)` before every utterance. Outputs saved as float WAV.
- Python 3.10.21, PyTorch 2.8.0+cu128, transformers 4.46.3, RTX 5070 Ti Laptop GPU (12 GB).

## Acoustic measures (`acoustics.py`)

Original and anonymized frames are matched by position: the anonymized frame grid is linearly
stretched onto the original's. This is exact for McAdams, and Seed-VC keeps the duration with
style conversion off.

- `f0_shift_st`: median F0 change in semitones over frames voiced in both (pYIN, 50–500 Hz)
- `f0_corr`: correlation of log-F0 over the same frames, i.e. whether the intonation shape is kept
- `env_lsd_db`: RMS dB difference between the gain-normalized order-20 LPC envelopes 1/|A| of
  20 ms frames, averaged over frames within 40 dB of the loudest original frame
