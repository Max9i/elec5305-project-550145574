# Python

Evaluation (attackers and ASR) and, later, the Seed-VC runs.

## Environment

Conda env `py5305` (Python 3.12). The RTX 50-series GPU needs a CUDA 12.8 build of PyTorch:

```bash
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r python/requirements.txt
```

Seed-VC will get its own env in M4, because its pinned dependencies clash with this one.

## Scripts

| Script | Purpose |
|---|---|
| `make_lists.py` | builds `data/lists/` once (see `data/README.md`) |
| `evaluate.py <condition> [wav_dir]` | ECAPA EER, WavLM-SV EER and ASR WER for one condition |
| `test_eer.py` | self-check of the EER function |

`wav_dir` holds one `<utt>.wav` per trial utterance, at any sample rate (everything is resampled to
16 kHz). Without `wav_dir`, the original LibriSpeech audio is used.

Outputs:

- `results/summary.csv`: one row per condition
- `results/scores/<condition>.csv`: cosine score of every pair for both attackers
- `results/asr/<condition>.csv`: ASR hypothesis and per-utterance WER
- `results/figures/scores_<condition>.png`: genuine vs. impostor score histograms

## Models (pretrained, frozen)

| Role | Model |
|---|---|
| Attacker A | `speechbrain/spkrec-ecapa-voxceleb` |
| Attacker B | `microsoft/wavlm-base-plus-sv` |
| ASR | `facebook/wav2vec2-large-960h-lv60-self`, greedy CTC decoding. Its output is already in LibriSpeech's upper-case transcript format, so no text normalisation is needed |

Speaker model = mean of the unit-length enrollment embeddings. Score = cosine similarity between
the trial embedding and the speaker model.
