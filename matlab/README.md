# MATLAB

Classical LPC/McAdams anonymiser (M2). Needs MATLAB R2026a and the Signal Processing Toolbox (`lpc`).

| File | Purpose |
|---|---|
| `mcadams_anon.m` | `y = mcadams_anon(x, fs, alpha)`: framing, LPC, residual, pole warping, resynthesis, overlap-add |
| `mcadams_warp.m` | `aNew = mcadams_warp(a, alpha)`: moves the complex LPC poles from angle φ to φ^α, keeping their radius |
| `test_mcadams.m` | self-check: radii kept, angles mapped to φ^α, α = 1 reconstructs the input |
| `run_mcadams.m` | anonymises every trial utterance for α = 0.9, 0.8, 0.7, 0.6 → `data/anon/mcadams_<α>/<utt>.wav` and `results/timing_mcadams.csv` |
| `demo_mcadams.m` | example analysis of one utterance → `samples/<utt>_*.wav` and `results/figures/mcadams_<utt>.png` |

Run from the repository root:

```bash
matlab -batch "cd matlab; test_mcadams"
matlab -batch "cd matlab; run_mcadams"
matlab -batch "cd matlab; demo_mcadams('121-127105-0006')"
```

Then evaluate each α, e.g. `python python/evaluate.py mcadams_0.8 data/anon/mcadams_0.8`.

## Implementation notes

- 20 ms frames, 10 ms hop and LPC order 20 (autocorrelation method), as in the VoicePrivacy McAdams
  baseline.
- √Hann analysis and synthesis windows. Their product is a periodic Hann window, which overlap-adds
  to exactly 1 at a 50 % hop. The signal is zero-padded by one hop at the start, so the first
  samples also lie in two frames. α = 1 therefore reconstructs the input up to rounding error
  (SNR 234 dB in `test_mcadams`).
- The conjugate poles are rebuilt from the transformed upper-half poles instead of being paired by
  position.
- All-zero frames have no LPC model and are left silent.
- The output is scaled to the input's peak, as in the reference implementation.
- Checked against a Python re-implementation of the VoicePrivacy reference (Burg LPC, symmetric
  Hann window): on both demo utterances the outputs match (waveform correlation ≥ 0.999, same crest
  factor).
- Speed: about 0.02–0.03 s of CPU time per second of audio (`results/timing_mcadams.csv`).
