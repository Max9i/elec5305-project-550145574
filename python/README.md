# Python

Evaluation (attackers, ASR, acoustic change) and the Seed-VC runs.

## Environments

Two conda envs. The RTX 50-series GPU needs CUDA 12.8 builds of PyTorch in both.

| Env | Python | Used by | Setup |
|---|---|---|---|
| `py5305` | 3.12 | everything except `run_seedvc.py` | `pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128` then `pip install -r python/requirements.txt` |
| `seedvc` | 3.10 | `run_seedvc.py` | `git clone https://github.com/Plachtaa/seed-vc third_party/seed-vc` (checkout `51383ef`), same PyTorch command, then `pip install -r python/requirements-seedvc.txt` |

Seed-VC's own requirements pin PyTorch 2.4.0 (cu121), which cannot run on this GPU, and other
versions that clash with `py5305`. That is why it has its own env.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `make_lists.py` | py5305 | builds `data/lists/` once (see `data/README.md`) |
| `run_seedvc.py <cfg>` | seedvc | Seed-VC V2 anonymisation-only for every trial → `data/anon/seedvc_cfg<cfg>/`, `results/timing_seedvc.csv` |
| `evaluate.py <condition> [wav_dir]` | py5305 | ECAPA EER, WavLM-SV EER and ASR WER for one condition |
| `acoustics.py <condition> <wav_dir>` | py5305 | F0 shift, F0 correlation and LPC-envelope distance vs. the original |
| `test_eer.py`, `test_acoustics.py` | py5305 | self-checks of the EER function and the acoustic measures |

`wav_dir` holds one `<utt>.wav` per trial utterance, at any sample rate (everything is resampled to
16 kHz). Without `wav_dir`, `evaluate.py` uses the original LibriSpeech audio.

Outputs:

- `results/summary.csv`: EERs and WER, one row per condition
- `results/acoustics.csv`: mean acoustic change, one row per condition (`results/acoustics/<condition>.csv` per utterance)
- `results/scores/<condition>.csv`: cosine score of every pair for both attackers
- `results/asr/<condition>.csv`: ASR hypothesis and per-utterance WER
- `results/figures/scores_<condition>.png`: genuine vs. impostor score histograms
- `results/timing_mcadams.csv`, `results/timing_seedvc.csv`: processing time per condition

## Evaluation models (pretrained, frozen)

| Role | Model | Revision |
|---|---|---|
| Attacker A | `speechbrain/spkrec-ecapa-voxceleb` | `0f99f2d` |
| Attacker B | `microsoft/wavlm-base-plus-sv` | `feb593a` |
| ASR | `facebook/wav2vec2-large-960h-lv60-self`, greedy CTC decoding. Its output is already in LibriSpeech's upper-case transcript format, so no text normalisation is needed | `54074b1` |

Speaker model = mean of the unit-length enrollment embeddings. Score = cosine similarity between
the trial embedding and the speaker model. Every waveform is scaled to the same RMS level before
scoring, because WavLM-SV's feature extractor does not normalise its input.

## Acoustic measures (`acoustics.py`)

Original and anonymised frames are matched by position: the anonymised frame grid is linearly
stretched onto the original's. This is exact for McAdams, and Seed-VC keeps the duration with
style conversion off.

- `f0_shift_st`: median F0 change in semitones over frames voiced in both (pYIN, 50–500 Hz)
- `f0_corr`: correlation of log-F0 over the same frames, i.e. whether the intonation shape is kept
- `env_lsd_db`: RMS dB difference between the gain-normalised order-20 LPC envelopes 1/|A| of
  20 ms frames, averaged over frames within 40 dB of the loudest original frame
