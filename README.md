# Privacy–Utility and Attacker Robustness in Classical and Neural Voice Anonymisation

ELEC5305 Project — Shuhuai Wang (SID 550145574) · GitHub: `Max9i/elec5305-project-550145574`

This project compares a classical LPC/McAdams anonymiser (MATLAB) with
pretrained zero-shot neural voice conversion (Seed-VC, Python). It does not
ask which anonymiser is "best". It asks **whether the privacy each one appears
to give still holds when a different speaker-verification attacker is used**,
and how much linguistic and acoustic utility is lost to get that privacy.

> The research question was revised after proposal feedback (see
> [Change log](#change-log)). The original submitted proposal is kept unchanged
> in `ELEC5305_Project_Proposal.pdf`.

---

## Current status

| Milestone | Status |
|---|---|
| M0 Repo setup, revised research question, README | ✅ done |
| M1 Evaluation dataset + original-speech attacker baseline | ✅ done |
| M2 MATLAB LPC/McAdams anonymiser | ⬜ next |
| M3 Classical privacy–utility trade-off | ⬜ |
| M4 Seed-VC anonymisation-only mode | ⬜ |
| M5 Attacker comparison (main result) | ⬜ |
| M6 Privacy-failure analysis | ⬜ |
| M7 One optional extension | ⬜ |
| M8 Report, figures, audio demos, GitHub Pages | ⬜ |

**Next step:** M2. Implement `matlab/mcadams_anon.m` (interface, pitfalls
and the α = 1 acceptance test are in `matlab/README.md`), then evaluate
each α with `python/evaluate.py`. In parallel, set up Seed-VC in its own
environment (M4).

**Results so far** (`results/summary.csv`; 400 genuine + 7600 impostor pairs)

| Condition | ECAPA EER (%) | WavLM-SV EER (%) | WER (%) |
|---|---|---|---|
| original | 0.25 | 3.26 | 1.73 |

On original speech both attackers clearly separate genuine from impostor
trials (`results/figures/scores_original.png`). ECAPA is the stronger one
here (d′ 7.3 vs. 2.6).

Code is pushed as each milestone is finished, not saved up for Weeks 12–13.

---

## 1. Research questions

**Main question**

> How robust are the privacy–utility trade-offs of classical LPC-based and
> modern zero-shot neural voice anonymisation when evaluated using different
> speaker-verification attackers?

**Secondary question**

> Which acoustic changes remove speaker identity while preserving linguistic
> and paralinguistic information, and which apparent privacy gains disappear
> when a stronger or different speaker representation is used?

**Why it matters.** Privacy is not a property of the anonymised waveform
alone. It also depends on who is attacking and which speaker representation
they use. In the VoicePrivacy 2025 Attacker Challenge, stronger attackers
substantially reduced the apparent privacy of several anonymisation systems
and changed their rankings. This project is an ELEC5305-scale version of the
same question.

**Possible outcomes (any of these is a valid result)**
- One anonymiser looks highly private against ECAPA but is still identifiable
  to WavLM-SV.
- Stronger McAdams modification gives more robust privacy but increasingly
  damages speech.
- Seed-VC gives good apparent anonymity and intelligibility but leaves some
  speakers much more identifiable than others.
- The privacy–utility ranking changes when the attacker changes.

---

## 2. Threat model

| Item | Definition in this project |
|---|---|
| Goal of the user | Share speech so the words (and ideally prosody/emotion) stay usable but the speaker cannot be re-identified |
| Attacker | Has **original** (non-anonymised) enrollment speech of candidate speakers, and tries to link an anonymised trial utterance to one of them using a pretrained speaker-verification model |
| Attacker knowledge (core) | **Ignorant attacker:** does not know about or adapt to the anonymiser. Enrollment = original speech, trial = anonymised speech |
| Attacker knowledge (optional) | **Semi-informed attacker:** knows the anonymiser, anonymises its own enrollment data the same way (Extension D) |
| Attackers used | A = ECAPA-TDNN (SpeechBrain), B = WavLM Base Plus SV (Microsoft) |
| Privacy metric | Equal Error Rate (EER). Higher EER = harder for the attacker |
| Utility metric | ASR WER increase (primary), plus F0/duration/energy/spectral preservation |

**Scope of the claim.** If both attackers fail, the conclusion is only that
*privacy was robust across the investigated attacker models*. It is **not**
that the speaker is anonymous against every possible attacker.

---

## 3. System overview

```
LibriSpeech subset ──► fixed enrollment / trial lists (saved once, reused everywhere)
        │
        ├──► original trial speech ───────────────────────────┐  (baseline)
        ├──► LPC/McAdams anonymiser (MATLAB), α = weak…strong ─┤
        └──► Seed-VC anonymisation-only (Python), 1 control axis┤
                                                              ▼
          ┌─────────────── Privacy ───────────────┐   ┌──────── Utility ────────┐
          │ Attacker A: ECAPA-TDNN  → EER          │   │ Fixed ASR → WER, ΔWER   │
          │ Attacker B: WavLM-SV    → EER          │   │ F0, duration, energy,   │
          │ per-speaker scores                     │   │ spectrogram, LPC envelope│
          └────────────────────────────────────────┘   │ processing time         │
                                                       └─────────────────────────┘
                              ▼
     Privacy–utility frontiers (one per attacker) + attacker-disagreement analysis
```

Each anonymiser configuration becomes **one point in privacy–utility space**,
evaluated by **both** attackers.

---

## 4. Anonymisers

### 4.1 Classical: LPC / McAdams (MATLAB) — student-implemented

This is the interpretable system and links directly to ELEC5305 material
(source–filter model, LPC, vocal-tract poles, frame processing, resynthesis).

**Pipeline**

```
speech → framing + window → LPC analysis (a_k) → residual e[n] = A(z)·s[n]
       → poles of 1/A(z) → McAdams pole-angle warping → new A'(z)
       → resynthesis s'[n] = e[n] / A'(z) → overlap-add
```

**McAdams transformation.** For each complex LPC pole `r·e^{jφ}` with
`0 < φ < π`, keep the radius `r` and change the angle to `φ' = φ^α` (φ in
radians). The conjugate pole is mirrored. Real poles are left unchanged.

- `α = 1` → no change.
- Moving `α` away from 1 moves the formants, which changes the estimated
  **vocal-tract spectral envelope** (speaker-specific timbre).
- The **excitation/residual** (and so mostly F0), the timing and the temporal
  structure are kept. The report must explain these physical effects rather
  than treat McAdams as a black-box function.

**Planned settings** (initial values, to confirm once the code works)

| Parameter | Initial value | Note |
|---|---|---|
| Sampling rate | 16 kHz | LibriSpeech native |
| Frame / hop | 20 ms / 10 ms, Hann, overlap-add | as in the VoicePrivacy McAdams baseline |
| LPC order | 20 | |
| McAdams α | 0.9 (weak), 0.8 (moderate), 0.7 (strong), optionally 0.6 | 3–5 values total, **no large parameter search** |

Per α, record: ECAPA EER, WavLM EER, WER, F0 change, spectral-envelope change,
processing time.

### 4.2 Neural: Seed-VC (Python) — pretrained, no training

- Official repo: <https://github.com/Plachtaa/seed-vc> (the link in the feedback, `guangzhouda/seed-vc`, is a fork)
- Paper: S. Liu, "Zero-shot Voice Conversion with Diffusion Transformers," arXiv:2411.09943, 2024
- **Do not train or fine-tune.** Seed-VC is used as an anonymisation tool. The
  contribution here is the controlled evaluation.

**Core mode: anonymisation-only (V2)**

`inference_v2.py --anonymization-only true` ignores the reference audio and
converts the source towards an "average" voice. This gives a clean neural
baseline with no target-speaker choices involved.

**One controlled axis only.** V2 exposes `--intelligibility-cfg-rate`,
`--similarity-cfg-rate` (default 0.7 each) and optional style conversion. Pick
**one** to vary (e.g. low / medium / high) and only if it is meaningful in
anonymisation-only mode. Keep all the others fixed.

**Output checks before evaluation:** sampling rate (resample everything to
16 kHz before scoring), duration vs. original, clipping, reproducibility (same
seed → same output), listening check.

**Reproducibility record** (fill in when installed; Seed-VC has changed a lot
since the 2024 paper):

| Item | Value |
|---|---|
| Repo commit / release | TBD |
| Model checkpoint | TBD |
| Config file | TBD |
| Random seed | TBD |
| Inference command + all parameters | TBD |
| Python / PyTorch / CUDA versions | TBD |
| GPU | TBD |

**Target-conditioned conversion is secondary (Extension A only).** Converting
towards a specific real person is *not* automatically anonymisation, because
it can add target similarity and impersonation risk and changes linkability.
If it is used:
- use only public research-dataset speakers, described as experimental references;
- keep a **separate target-speaker pool** that does not overlap the evaluation speakers;
- never allow target = source.

---

## 5. Attackers (privacy evaluation)

| | Attacker A | Attacker B |
|---|---|---|
| Model | SpeechBrain ECAPA-TDNN | Microsoft WavLM Base Plus SV |
| Link | <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb> | <https://huggingface.co/microsoft/wavlm-base-plus-sv> |
| Representation | Conventional supervised speaker embedding (VoxCeleb) | Self-supervised representation with speaker-aware pretraining, x-vector head |
| Training | none (pretrained, frozen) | none (pretrained, frozen) |

Using two very different representations is what makes the robustness
question testable.

### Protocol

1. **Baseline first:** original enrollment → original trial. Both attackers
   should give clearly separated genuine and impostor scores and a low EER.
   *Do not anonymise anything until this works.*
2. **Ignorant attacker:** original enrollment → anonymised trial, for every
   anonymiser configuration.
3. Scoring: cosine similarity between embeddings, then EER from the genuine
   and impostor score sets.
4. Report **EER** as the main privacy metric. Cosine-similarity plots are
   diagnostics only. Raw cosine values are not comparable across attackers
   (WavLM-SV scores sit between about 0.4 and 1.0, ECAPA's between about
   −0.2 and 1.0), so compare attackers by EER.
5. Report **per-speaker results** as well as the average: mean genuine score
   before/after anonymisation and the privacy improvement for each speaker.

**EER:** the operating point where the false-acceptance rate equals the
false-rejection rate. ~0% means the speaker is easy to verify. ~50% means the
attacker does no better than chance.

---

## 6. Utility evaluation

**Primary: WER**
- One fixed pretrained ASR for every condition:
  `facebook/wav2vec2-large-960h-lv60-self` (wav2vec2-large fine-tuned on
  LibriSpeech 960 h, which does not include test-clean), greedy CTC decoding,
  no language model. No ASR training.
- For each trial utterance compare: reference transcript, ASR on original,
  ASR on anonymised.
- Key quantity: **ΔWER = WER(anonymised) − WER(original)**, the extra error
  introduced by anonymisation.

**Acoustic / prosodic preservation (ELEC5305 analysis)**
- F0 contour: original vs. anonymised. McAdams is expected to keep F0 mostly
  intact, while Seed-VC may change more timbre and style cues. This is a key contrast.
- Duration, energy contour.
- Spectrogram and LPC spectral envelope (original vs. McAdams-modified).
- Formant structure (optional).
- Processing time per second of audio, for each method.

**Emotion (SER UAR)**: optional only (Extension C).

---

## 7. Dataset and trial protocol

- **Dataset:** LibriSpeech `test-clean`. It has many speakers, transcripts
  are available, and VoicePrivacy uses it for both ASV and ASR evaluation.
- **Speakers:** all **40** test-clean speakers (20 F / 20 M). The plan said
  10–20, but compute is not a constraint here and the lists are fixed once,
  so the full set was used for more stable EERs and per-speaker results.
- **Enrollment vs. trial:** per speaker, **5 enrollment** and **10 trial**
  utterances (disjoint, 4–15 s, drawn with seed 5305). Enrollment utterances
  define the original identity and are never anonymised. Trial utterances
  (400 in total, 0.87 h) are anonymised and scored. An anonymised file is
  never compared with its own original.
- **Trials:** every trial utterance is scored against every same-gender
  speaker model, giving **400 genuine + 7600 impostor** pairs. Impostors are
  same gender, following VoicePrivacy practice.
- **Session caveat:** a speaker's enrollment and trial utterances come from
  the same chapters (recording sessions), so the attacker can also use
  session/channel similarity. This favours the attacker, so it is
  conservative for privacy, but it may help against McAdams (which keeps the
  channel) more than against Seed-VC (which resynthesises everything).
- **Fixed lists:** `speakers.csv`, `enroll.csv`, `trials.csv` and
  `pairs.csv` were generated **once** by `python/make_lists.py`, saved in
  `data/lists/` and **committed**. Every anonymiser and every attacker uses
  exactly the same lists.
- Raw audio is not committed. `data/README.md` explains how to download it
  and what each list contains.

---

## 8. Planned figures

1. **Privacy–utility frontier**: x = ΔWER, y = EER. Points: original,
   each McAdams α, each Seed-VC setting. **One plot per attacker**
   (ECAPA, WavLM). Different shapes or rankings between the two plots are
   direct evidence for the main question.
2. **Attacker disagreement**: x = ECAPA EER, y = WavLM EER for every
   configuration. Points that fall away from a common trend show methods that
   exploit the weakness of one representation.
3. **Acoustic change vs. privacy (McAdams)**: amount of spectral-envelope
   change vs. EER, linking the DSP mechanism to the privacy result.
4. **Example analysis**: for representative utterances, waveform/spectrogram,
   LPC envelope (original vs. McAdams), Seed-VC spectrogram, F0 contour,
   similarity scores, transcript/WER.
5. **Speaker-level risk**: privacy improvement per speaker, highlighting
   speakers protected by both systems, speakers still identifiable, and
   ECAPA/WavLM disagreement.

---

## 9. Milestones (detailed)

**M1 – Evaluation dataset + baseline**
- [x] Download LibriSpeech test-clean, choose speakers
- [x] Generate and commit enrollment/trial/impostor lists (fixed seed)
- [x] ECAPA embeddings + EER on original speech
- [x] WavLM-SV embeddings + EER on original speech
- [x] ASR WER on original speech
- [x] Check genuine and impostor scores are clearly separated

**M2 – MATLAB LPC/McAdams anonymiser**
- [ ] Framing → LPC → poles → McAdams warping → resynthesis → overlap-add
- [ ] Verify α = 1 reconstructs the input (sanity check)
- [ ] Waveform, spectrogram, LPC envelope plots, and original/anonymised audio
- [ ] Batch-process trial list for 3–5 α values

**M3 – Classical privacy–utility trade-off**
- [ ] For each α: ECAPA EER, WavLM EER, WER/ΔWER, F0 change, processing time

**M4 – Seed-VC anonymisation-only**
- [ ] Install, fill in reproducibility record (section 4.2)
- [ ] Process the same trial utterances
- [ ] Check sampling rate, duration, clipping, reproducibility, quality
- [ ] Same evaluation as M3

**M5 – Attacker comparison (main result)**
- [ ] Figures 1 and 2
- [ ] Do ECAPA and WavLM give the same privacy conclusion and the same ranking?

**M6 – Privacy-failure analysis**
- [ ] Find examples: both attackers fooled / both still identify / only ECAPA fooled / only WavLM fooled
- [ ] Analyse F0, spectral envelope, spectrogram, prosody, speaking rate for those cases
- [ ] Figures 3, 4, 5

**M7 – One optional extension** (section 10)

**M8 – Final deliverables**
- [ ] Report, selected audio samples in `samples/`, figures in `results/`, GitHub Pages site

---

## 10. Optional extensions (choose **one** only, and only after M6)

| Option | Idea |
|---|---|
| A. Target selection | Seed-VC: anonymous-average vs. fixed target vs. random target (or acoustically similar vs. different targets). How does pseudo-speaker choice affect privacy, utility and linkability? |
| B. Linkability | Are anonymised utterances from the same original speaker still more similar to each other than to other speakers' utterances? |
| C. Emotion preservation | SER UAR before/after anonymisation on a small IEMOCAP subset with one fixed pretrained SER model |
| D. Semi-informed attacker | Reduced version: the attacker anonymises its enrollment data with the same system |

## 11. Out of scope

Not done in this project: training or fine-tuning Seed-VC, new diffusion
models, training ECAPA/WavLM/ASR, reproducing the full VoicePrivacy Challenge
(large datasets, anonymised-domain ASV retraining), building a large dataset,
implementing every privacy metric, or proving absolute anonymity.

Core scope = **2 anonymisers × several configurations × 2 pretrained attackers
× a manageable LibriSpeech speaker subset**.

---

## 12. Repository structure

```
├── README.md                       this file (plan + status)
├── ELEC5305_Project_Proposal.pdf   original submitted proposal (unchanged)
├── matlab/     LPC/McAdams anonymiser
├── python/     data lists, Seed-VC runs, ECAPA/WavLM/ASR evaluation, EER, plots
├── data/       dataset instructions + committed trial lists (data/lists/), no raw audio
├── results/    metric tables (CSV) and figures
└── samples/    selected short audio examples (original vs. anonymised)
```

Run instructions live next to the code: `data/README.md` (dataset and
lists), `python/README.md` (evaluation environment and scripts, with
`python/requirements.txt`) and `matlab/README.md` (McAdams interface and
checks).

---

## 13. Resources

| Resource | Link |
|---|---|
| VoicePrivacy Challenge | <https://www.voiceprivacychallenge.org/> |
| VoicePrivacy 2024 code (anonymisation baselines, ASV/ASR/SER evaluation; methodological reference) | <https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024> |
| First VoicePrivacy Attacker Challenge (ICASSP 2025) | <https://www.voiceprivacychallenge.org/attacker/> |
| 2026 analysis: privacy attacks on voice anonymization systems | <https://www.sciencedirect.com/science/article/pii/S0885230826000999> |
| Seed-VC (official) | <https://github.com/Plachtaa/seed-vc> |
| ECAPA-TDNN attacker | <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb> |
| WavLM-SV attacker | <https://huggingface.co/microsoft/wavlm-base-plus-sv> |

### References (from proposal)

1. S. Liu, "Zero-shot Voice Conversion with Diffusion Transformers," arXiv:2411.09943, 2024.
2. N. Tomashenko et al., "The VoicePrivacy 2020 Challenge: Results and findings," *Computer Speech & Language*, vol. 74, 101362, 2022.
3. J. Patino, N. Tomashenko, M. Todisco, A. Nautsch, N. Evans, "Speaker Anonymisation Using the McAdams Coefficient," *Proc. Interspeech*, pp. 1099–1103, 2021.
4. F. Fang et al., "Speaker Anonymization Using X-vector and Neural Waveform Models," *Proc. SSW 10*, pp. 155–160, 2019.
5. B. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," *Proc. Interspeech*, pp. 3830–3834, 2020.

### Literature still to add (for the report)

The literature review should lead to *"How much privacy does an anonymisation
system really provide when the attacker changes?"* Topics to cover:
- [ ] VoicePrivacy 2024 Challenge (evaluation plan / results)
- [ ] 2025 VoicePrivacy Attacker Challenge
- [ ] Attacker-dependent privacy evaluation (ignorant vs. semi-informed attackers)
- [ ] Linkability
- [ ] Speaker-level privacy risk
- [ ] WavLM and self-supervised speaker representations
- [ ] Preservation of emotional/prosodic utility
- [ ] Modern neural anonymisation

---

## Change log

- **2026-08-30**: Repo initialised from the proposal (original question: does
  LPC/McAdams or Seed-VC give the better privacy–intelligibility trade-off?).
- **2026-10-01**: Revised after proposal feedback:
  - The central question is now **robustness of the privacy claim across
    attackers**, and the title was changed to match.
  - Added a second attacker (WavLM-SV), EER as the main privacy metric, an
    explicit threat model, and a fixed enrollment/trial protocol.
  - Seed-VC now starts in anonymisation-only mode, with target-speaker
    conversion moved to an optional extension.
  - Added ΔWER and acoustic/prosodic preservation analysis, per-speaker
    results, and five planned figures.
  - Milestones reordered so the attacker baseline comes before any anonymisation.
- **2026-10-01**: M1 done.
  - Fixed lists for all 40 test-clean speakers (instead of the planned
    10–20): 5 enrollment + 10 trial utterances each, 400 genuine + 7600
    same-gender impostor pairs.
  - ASR fixed to `facebook/wav2vec2-large-960h-lv60-self`.
  - Original-speech baseline: ECAPA EER 0.25 %, WavLM-SV EER 3.26 %, WER
    1.73 %. A rerun gave identical scores.
  - `.gitignore` now keeps `data/lists/` and `results/` in the repository.
