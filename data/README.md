# Data

Raw and anonymised audio stay local (gitignored). Only `lists/` is committed.

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
| `enroll.csv` | 5 utterances per speaker (200 total). Original speech only, never anonymised |
| `trials.csv` | 10 other utterances per speaker (400 total, 0.87 h) with reference transcripts. These are anonymised |
| `pairs.csv` | every trial against every same-sex speaker model: 400 target + 7600 impostor pairs |

Only utterances of 4–15 s are used. A speaker's enrollment and trial utterances are disjoint
but come from the same chapters (recording sessions), so the attacker also benefits from
session/channel similarity. Keep this in mind when interpreting the results.

## Local-only folders

| Folder | Content |
|---|---|
| `raw/` | LibriSpeech |
| `anon/<condition>/<utt>.wav` | anonymised trial audio, one file per line of `trials.csv` |
| `models/` | local copy of the SpeechBrain ECAPA model |
