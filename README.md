# Classical vs. Neural Voice Anonymization Under Different Attackers

ELEC5305 project, Shuhuai Wang (SID 550145574).

I compare two voice anonymizers on LibriSpeech test-clean:

- **McAdams**: a classical LPC anonymizer that I wrote in MATLAB. It moves the
  formants by warping the LPC pole angles (α = 0.9, 0.8, 0.7, 0.6).
- **Seed-VC**: a pretrained neural voice converter, used in its
  anonymization-only mode (no training).

The question is whether the privacy of each anonymizer still holds when the
attacker changes. Privacy is measured by the EER of two speaker-verification
attackers (ECAPA-TDNN and WavLM-SV). Utility is measured by ASR word error
rate (WER).

The full write-up is in [`report/main.pdf`](report/main.pdf).

## Main results

EER in % (higher = more private). "Informed" is an attacker that also
anonymizes its enrollment speech; the column shows the lower EER of the two
attackers.

| Condition | ECAPA | WavLM-SV | Informed | WER |
|---|---|---|---|---|
| original | 0.3 | 3.5 | – | 1.7 |
| McAdams α 0.8 | 8.0 | 17.5 | 1.8 | 2.1 |
| McAdams α 0.6 | 29.8 | 40.3 | 8.7 | 9.4 |
| Seed-VC CFG 0.7 | 43.8 | 48.0 | 39.3 | 2.4 |

- Both attackers rank the systems the same way: Seed-VC gives much more
  privacy than McAdams, at a similar WER.
- The amount of privacy McAdams seems to give depends on the attacker.
  WavLM-SV is fooled much more easily than ECAPA.
- Once the attacker anonymizes its own enrollment speech, most of McAdams'
  privacy is gone. Seed-VC keeps an EER near 40 %.

![Privacy vs. utility](results/figures/fig1_privacy_utility.png)

All numbers are in `results/` (`summary.csv`, `summary_lazy.csv`,
`eer_ci.csv`, ...). Listening examples are in `samples/`.

## How to run

Setup details for each part are in [`data/README.md`](data/README.md),
[`matlab/README.md`](matlab/README.md) and
[`python/README.md`](python/README.md). Short version, from the repo root:

```bash
# 1. data: download LibriSpeech test-clean into data/raw/
# 2. baseline on original speech
python python/evaluate.py original
# 3. McAdams (MATLAB), then evaluate one setting
matlab -batch "cd matlab; run_mcadams"
python python/evaluate.py mcadams_0.8 data/anon/mcadams_0.8
# 4. Seed-VC, then evaluate
python python/run_seedvc.py 0.7
python python/evaluate.py seedvc_cfg0.7 data/anon/seedvc_cfg0.7
# 5. figures
python python/plot_m5.py
python python/plot_m6.py
```

## Repository layout

```
matlab/    McAdams anonymizer and its self-test
python/    Seed-VC runs, attackers, ASR, acoustic measures, plots
data/      download instructions and the fixed trial lists
results/   CSV results and figures
samples/   original vs. anonymized audio examples
report/    final report (PDF)
```

## Main references

- S. Liu, "Zero-shot Voice Conversion with Diffusion Transformers," arXiv:2411.09943, 2024. Code: <https://github.com/Plachtaa/seed-vc>
- J. Patino et al., "Speaker Anonymisation Using the McAdams Coefficient," Interspeech 2021.
- VoicePrivacy Challenge: <https://www.voiceprivacychallenge.org/>

The original proposal is kept in `ELEC5305_Project_Proposal.pdf`.
