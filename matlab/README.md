# MATLAB

Classical LPC/McAdams anonymiser (M2). Needs MATLAB R2026a and the Signal Processing Toolbox (`lpc`).

## Interface expected by the Python evaluation

- `y = mcadams_anon(x, fs, alpha)`: `x` is mono speech (column vector) at `fs` = 16000 Hz, `alpha` is
  the McAdams coefficient (1 = no change). `y` has the same length as `x`.
- Batch run: for each α in {0.9, 0.8, 0.7} (optionally 0.6), read every utterance in
  `data/lists/trials.csv` from `data/raw/LibriSpeech/test-clean/<spk>/<chapter>/<utt>.flac` and write
  `data/anon/mcadams_<α>/<utt>.wav` at 16 kHz. Record the total processing time and audio duration
  per α for the processing-time comparison.
- Then evaluate, e.g. `python python/evaluate.py mcadams_0.8 data/anon/mcadams_0.8`.

## Algorithm (README §4.1)

Per frame (20 ms window, 10 ms hop, LPC order 20):

1. windowed frame → `a = lpc(frame, 20)`
2. residual `e = filter(a, 1, frame)`
3. poles `r = roots(a)`
4. complex poles with 0 < φ < π: keep |r| and set φ' = φ^α, clamped to (0, π). Rebuild the
   conjugates from the new upper-half poles. Leave real poles unchanged.
5. `a_new = real(poly(r_new))`
6. resynthesis `filter(1, a_new, e)` → synthesis window → overlap-add

## Pitfalls

- **α = 1 must give back the input** (see the acceptance test below). If it does not, the bug is in
  the windowing or overlap-add, not in McAdams. Use √Hann as both the analysis and the synthesis
  window, scaled so that the overlapped product of the two sums to 1 at a 10 ms hop (this is what
  the VoicePrivacy baseline does).
- Pair the conjugate poles explicitly: take the poles with `imag(r) > 0`, transform them, and append
  their conjugates. Do not assume `roots` returns each pair next to each other.
- φ is in radians, so φ^α has a fixed point at φ = 1 rad, i.e. fs/2π ≈ 2.55 kHz. With α < 1,
  formants below about 2.5 kHz move up and formants above it move down. This is worth explaining
  in the report.
- Silent frames make `lpc` return NaN. Add a tiny eps to the frame or skip it.
- `audiowrite` clips at ±1. Scale the output to the input's peak before writing.

## Acceptance test

```matlab
[x, fs] = audioread('data/raw/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac');
y = mcadams_anon(x, fs, 1);
k = 400:numel(x) - 400;  % ignore the edge frames
fprintf('alpha = 1: SNR = %.1f dB\n', 10*log10(sum(x(k).^2) / sum((x(k) - y(k)).^2)))  % expect > 30 dB
```
