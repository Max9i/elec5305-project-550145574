# Data

Raw and anonymized audio stay local (gitignored). Only `lists/` is committed.

## LibriSpeech test-clean

```bash
mkdir -p data/raw && cd data/raw
curl -LO https://www.openslr.org/resources/12/test-clean.tar.gz   # 346 MB, md5 32fa31d27d2e1cad72775fee3f4849a9
tar -xzf test-clean.tar.gz                                         # -> data/raw/LibriSpeech/
```

## Fixed lists (`lists/`, committed)

Generated once with `python python/make_lists.py` (seed 5305). Every result depends on these
lists, so do not regenerate them unless the protocol is changed on purpose.

| File | Content |
|---|---|
| `speakers.csv` | all 40 test-clean speakers (20 F, 20 M) |
| `enroll.csv` | 5 utterances per speaker (200 total). Original for the ignorant attacker; anonymized only for the informed attackers (M7) |
| `trials.csv` | 10 other utterances per speaker (400 total, 0.87 h) with reference transcripts. These are anonymized |
| `backend.csv` | M7 only: 20 utterances of each of the 40 dev-clean speakers (disjoint from test-clean), the attacker's own data for the informed backend. Needs `dev-clean.tar.gz` (337 MB, md5 42e2234ba48799c1f50f24a7926300a1) extracted next to test-clean |
| `pairs.csv` | every trial against every same-sex speaker model: 400 target + 7600 impostor pairs |

Only utterances of 4–15 s are used. A speaker's enrollment and trial utterances are disjoint
but come from the same chapters (recording sessions), so the attacker also benefits from
session/channel similarity. Keep this in mind when interpreting the results.

## Local-only folders

| Folder | Content |
|---|---|
| `raw/` | LibriSpeech test-clean and dev-clean |
| `anon/<condition>/<utt>.wav` | anonymized audio: every utterance of `trials.csv`, plus `enroll.csv` and `backend.csv` for M7 |
| `models/` | local copy of the SpeechBrain ECAPA model |
