"""M6 privacy-failure analysis: which trials and speakers stay identifiable, to which attacker, and why.

    python python/analyse_m6.py

Each genuine trial score is turned into a z-score against the impostor scores of the same condition,
attacker and sex (impostor pairs are same-sex, and raw cosines are not comparable across attackers).
A trial counts as identified when it scores above the 95th percentile of those impostor scores, i.e.
at a 5 % false-acceptance rate, where chance alone identifies 5 % of trials. Prints the statistics;
figures are made by plot_m6.py.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from acoustics import LIBRI, SR, f0

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ATTACKERS = ("ecapa", "wavlm")
CONDS = ["original", "mcadams_0.9", "mcadams_0.8", "mcadams_0.7", "mcadams_0.6",
         "seedvc_cfg0.0", "seedvc_cfg0.7", "seedvc_cfg1.0"]


def rho(x, y):
    """Spearman correlation over the pairs where both values exist."""
    return pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float)), method="spearman")


def trial_z(cond):
    """Per genuine trial: z-score and identified flag (FAR 5 %) for both attackers, per sex."""
    p = pd.read_csv(RES / "scores" / f"{cond}.csv", dtype={"trial": str, "model_spk": str})
    sex = pd.read_csv(ROOT / "data/lists/speakers.csv", dtype=str).set_index("spk").sex
    p["sex"] = p.model_spk.map(sex)
    out = []
    for _, g in p.groupby("sex"):
        imp, gen = g[g.target == 0], g[g.target == 1].set_index("trial")
        out.append(pd.DataFrame({k: v for a in ATTACKERS for k, v in
                                 [(f"z_{a}", (gen[a] - imp[a].mean()) / imp[a].std()),
                                  (f"hit_{a}", gen[a] > imp[a].quantile(0.95))]}))
    return pd.concat(out)


def far5_z(cond, attacker):
    """The identification threshold (95th impostor percentile) in z units, averaged over the sexes."""
    p = pd.read_csv(RES / "scores" / f"{cond}.csv", dtype={"trial": str, "model_spk": str})
    sex = pd.read_csv(ROOT / "data/lists/speakers.csv", dtype=str).set_index("spk").sex
    imp = p[p.target == 0].assign(sex=lambda d: d.model_spk.map(sex))
    return imp.groupby("sex")[attacker].apply(lambda s: (s.quantile(0.95) - s.mean()) / s.std()).mean()


def median_f0(utt):
    spk, chap, _ = utt.split("-")
    return np.nanmedian(f0(librosa.load(LIBRI / spk / chap / f"{utt}.flac", sr=SR)[0]))


def trials():
    """Trial list with speaking rate and the original median F0 (cached, pYIN takes a few minutes)."""
    tr = pd.read_csv(ROOT / "data/lists/trials.csv", dtype={"utt": str, "spk": str}).set_index("utt")
    tr["rate"] = tr.text.str.split().str.len() / tr.dur
    en = pd.read_csv(ROOT / "data/lists/enroll.csv", dtype=str)
    chapters = en.utt.str.split("-").str[1].groupby(en.spk).apply(set)
    tr["same_chapter"] = [u.split("-")[1] in chapters[s] for u, s in zip(tr.index, tr.spk)]
    cache = RES / "acoustics" / "original_f0.csv"
    if not cache.exists():
        with ProcessPoolExecutor(max_workers=8) as ex:
            f = list(ex.map(median_f0, tr.index, chunksize=4))
        pd.DataFrame({"utt": tr.index, "f0_hz": np.round(f, 1)}).to_csv(cache, index=False)
    return tr.join(pd.read_csv(cache, dtype={"utt": str}).set_index("utt"))


def table(cond):
    """Trials plus acoustic change and identifiability for one anonymised condition."""
    ac = pd.read_csv(RES / "acoustics" / f"{cond}.csv", dtype={"utt": str}).set_index("utt")
    d = trials().join(ac).join(trial_z(cond)).join(trial_z("original").add_suffix("_orig"))
    d["f0_anon_hz"] = d.f0_hz * 2 ** (d.f0_shift_st / 12)  # median output F0, from the median shift
    return d


def main():
    print("Identified trials at FAR 5 % (chance = 20 of 400 per attacker): both / only ECAPA / only WavLM-SV / neither")
    for c in CONDS:
        z = trial_z(c)
        e, w = z.hit_ecapa, z.hit_wavlm
        print(f"  {c:14s} {(e & w).sum():4d} {(e & ~w).sum():4d} {(~e & w).sum():4d} {(~e & ~w).sum():4d}")

    tr = trials()
    S = {c: trial_z(c).join(tr.spk).groupby("spk")[["z_ecapa", "z_wavlm"]].mean() for c in CONDS}
    print("\nSpeaker-level mean z (40 speakers), Spearman rho:")
    for c in CONDS[1:]:
        print(f"  {c:14s} ECAPA vs WavLM-SV {rho(S[c].z_ecapa, S[c].z_wavlm):+.2f}; vs original speech: "
              f"ECAPA {rho(S[c].z_ecapa, S['original'].z_ecapa):+.2f}, WavLM-SV {rho(S[c].z_wavlm, S['original'].z_wavlm):+.2f}")
    for a in ATTACKERS:
        print(f"  {a}: McAdams 0.8 vs 0.7 {rho(S['mcadams_0.8'][f'z_{a}'], S['mcadams_0.7'][f'z_{a}']):+.2f}, "
              f"McAdams 0.8 vs Seed-VC 0.7 {rho(S['mcadams_0.8'][f'z_{a}'], S['seedvc_cfg0.7'][f'z_{a}']):+.2f}")

    # typicality: how similar a speaker's original voice is to the other same-sex speakers (mean impostor score
    # of their model on original speech, z within sex) vs. how identifiable they stay after anonymisation
    o = pd.read_csv(RES / "scores/original.csv", dtype={"trial": str, "model_spk": str})
    sex = tr.groupby("spk").sex.first()
    typ = o[o.target == 0].groupby("model_spk")[list(ATTACKERS)].mean()
    typ = typ.groupby(sex).transform(lambda s: (s - s.mean()) / s.std())
    for c in ["mcadams_0.8", "mcadams_0.7", "seedvc_cfg0.7"]:
        cells = []
        for a in ATTACKERS:
            r = spearmanr(typ[a], S[c][f"z_{a}"].reindex(typ.index))
            cells.append(f"{a} {r.statistic:+.2f} (p {r.pvalue:.2f})")
        print(f"  typicality vs speaker risk, {c:14s} " + "   ".join(cells))

    # session cue: the 10 speakers with trials both from enrollment chapters and from other chapters
    mixed = tr.groupby("spk").same_chapter.transform("nunique") == 2
    print(f"\nSame-chapter minus other-chapter mean z, within the {tr[mixed].spk.nunique()} speakers that have both "
          f"({(mixed & ~tr.same_chapter).sum()} other-chapter trials); 'share' = that gap / the speakers' mean z")
    for c in CONDS:
        d = tr[mixed].join(trial_z(c))
        cells = []
        for a in ATTACKERS:
            gap = d.groupby(["spk", "same_chapter"])[f"z_{a}"].mean().unstack().pipe(lambda m: m[True] - m[False])
            cells.append(f"{a} {gap.mean():+.2f} (share {gap.mean() / d[f'z_{a}'].mean():.2f}, {(gap > 0).sum()}/{len(gap)} speakers > 0)")
        print(f"  {c:14s} " + "   ".join(cells))

    feats = ["dur", "rate", "f0_hz", "env_lsd_db", "voiced_kept", "f0_shift_st", "f0_corr"]
    for c in ["mcadams_0.8", "mcadams_0.7", "seedvc_cfg0.7"]:
        d = table(c)
        print(f"\n{c}: trial z vs features, Spearman rho (all / F / M)")
        for f in feats + ["abs_shift"]:
            if f == "abs_shift":
                d[f] = d.f0_shift_st.abs()
            cells = []
            for a in ATTACKERS:
                r = [rho(g[f], g[f"z_{a}"]) for g in (d, d[d.sex == "F"], d[d.sex == "M"])]
                cells.append(f"{a} {r[0]:+.2f} / {r[1]:+.2f} / {r[2]:+.2f}")
            print(f"  {f:12s} " + "   ".join(cells))
    d = table("mcadams_0.7")
    within = d.groupby("spk")[["voiced_kept", "z_ecapa"]].transform(lambda s: s - s.mean())
    between = d.groupby("spk")[["voiced_kept", "z_ecapa", "f0_hz"]].mean()
    print(f"\nMcAdams 0.7, voicing kept vs ECAPA z: within speakers {rho(within.voiced_kept, within.z_ecapa):+.2f}, "
          f"between speakers {rho(between.voiced_kept, between.z_ecapa):+.2f} "
          f"(voicing kept vs speaker F0 {rho(between.voiced_kept, between.f0_hz):+.2f})")

    d = table("seedvc_cfg0.7")
    print("\nSeed-VC 0.7 output median F0 by sex (Hz):", d.groupby("sex").f0_anon_hz.median().round(0).to_dict(),
          "| original:", d.groupby("sex").f0_hz.median().round(0).to_dict(),
          f"| rho(original F0, output F0) within F {rho(*d[d.sex == 'F'][['f0_hz', 'f0_anon_hz']].T.values):+.2f},"
          f" within M {rho(*d[d.sex == 'M'][['f0_hz', 'f0_anon_hz']].T.values):+.2f}")


if __name__ == "__main__":
    main()
