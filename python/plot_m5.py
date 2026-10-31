"""M5 attacker comparison: Figures 1 and 2, speaker-bootstrap EER intervals, ranking agreement.

    python python/plot_m5.py

Reads results/summary.csv and results/scores/<condition>.csv. Writes results/eer_ci.csv,
results/figures/fig1_privacy_utility.png and results/figures/fig2_attacker_disagreement.png, and
prints which condition pairs the two attackers rank differently.

95 % intervals come from resampling the trial speakers with replacement (20 F and 20 M each draw),
the same draws for every condition and attacker, so differences between conditions are paired.
"""
import itertools

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from evaluate import LISTS, NAMES, RES, eer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, ATTACKERS = 1000, ("ecapa", "wavlm")
SYSTEMS = [("McAdams", "#2a78d6", [(f"mcadams_{a}", f"α {a}") for a in ("0.9", "0.8", "0.7", "0.6")]),
           ("Seed-VC", "#eb6834", [(f"seedvc_cfg{c}", f"CFG {c}") for c in ("0.0", "0.7", "1.0")])]
INK, MUTED = "#52514e", "#898781"

summary = pd.read_csv(RES / "summary.csv").set_index("condition")
summary["dwer"] = summary.wer - summary.wer["original"]
conds = list(summary.index)
# M7: lazy-informed attacker (anonymized enrollment) and lazy-informed + adapted backend, if evaluated;
# bootstrapped with the same draws
read = lambda name: pd.read_csv(RES / name).set_index("condition") if (RES / name).exists() else None
lazy, informed = read("summary_lazy.csv"), read("summary_backend.csv")
points = pd.concat([summary, lazy, informed])[["ecapa_eer", "wavlm_eer"]]

# paired speaker bootstrap: one list of drawn trial speakers per draw, reused everywhere
spk = pd.read_csv(LISTS / "speakers.csv", dtype=str)
rng = np.random.default_rng(5305)
draws = [np.concatenate([rng.choice(g.spk, len(g)) for _, g in spk.groupby("sex")]) for _ in range(B)]
boot = {}
for c in points.index:
    p = pd.read_csv(RES / "scores" / f"{c}.csv", dtype={"trial": str, "model_spk": str})
    rows = p.groupby(p.trial.str.split("-").str[0]).indices  # trial speaker -> row numbers
    for a in ATTACKERS:
        s, t = p[a].to_numpy(), p.target.to_numpy()
        boot[c, a] = np.array([eer(s[i], t[i]) for i in (np.concatenate([rows[k] for k in d]) for d in draws)])
ci = pd.DataFrame([dict(condition=c, attacker=a, eer=points.loc[c, f"{a}_eer"],
                        lo=np.percentile(boot[c, a], 2.5), hi=np.percentile(boot[c, a], 97.5))
                   for c in points.index for a in ATTACKERS]).round(2)
ci.to_csv(RES / "eer_ci.csv", index=False)
ci = ci.set_index(["condition", "attacker"])

# attacker gap per condition: how much more privacy WavLM-SV reports than ECAPA, and how much of that
# goes beyond the gap already present on original speech (WavLM-SV is the weaker attacker there)
base = boot["original", "wavlm"] - boot["original", "ecapa"]
for c in conds:
    gap = boot[c, "wavlm"] - boot[c, "ecapa"]
    print(f"{c}: WavLM-SV minus ECAPA EER {summary.loc[c, 'wavlm_eer'] - summary.loc[c, 'ecapa_eer']:+.2f} "
          f"(95 % CI {np.percentile(gap, 2.5):+.2f} to {np.percentile(gap, 97.5):+.2f}); beyond the original-speech "
          f"gap: 95 % CI {np.percentile(gap - base, 2.5):+.2f} to {np.percentile(gap - base, 97.5):+.2f}")

# ranking agreement between the attackers
tau = kendalltau(summary.ecapa_eer, summary.wavlm_eer)
print(f"Kendall tau between ECAPA and WavLM-SV rankings of {len(conds)} conditions: {tau.statistic:.2f}")
for c1, c2 in itertools.combinations(conds, 2):
    d = {a: boot[c1, a] - boot[c2, a] for a in ATTACKERS}
    point = {a: summary.loc[c1, f"{a}_eer"] - summary.loc[c2, f"{a}_eer"] for a in ATTACKERS}
    if np.sign(point["ecapa"]) != np.sign(point["wavlm"]):
        share = {a: max(np.mean(d[a] > 0), np.mean(d[a] < 0)) for a in ATTACKERS}
        print(f"order differs: {c1} vs {c2}: ECAPA {point['ecapa']:+.2f} ({share['ecapa']:.0%} of draws agree), "
              f"WavLM-SV {point['wavlm']:+.2f} ({share['wavlm']:.0%} of draws agree)")

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "font.size": 9})


def point(ax, x, y, xerr, yerr, color, label, offset):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", ms=6, color=color, ecolor=color, elinewidth=1, capsize=0,
                mec="#fcfcfb", mew=1.5, zorder=3)
    ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8, color=INK,
                ha="left" if offset[0] >= 0 else "right", va="center")


def yerr(c, a):
    r = ci.loc[(c, a)]
    return [[r.eer - r.lo], [r.hi - r.eer]]


# Figure 1: privacy-utility frontier, one panel per attacker
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True, facecolor="#fcfcfb")
for ax, a in zip(axes, ATTACKERS):
    ax.axhline(50, ls="--", lw=1, color=MUTED)
    ax.text(8.2, 51, "chance", color=MUTED, fontsize=8, ha="right")
    point(ax, 0, summary.loc["original", f"{a}_eer"], None, yerr("original", a), INK, "original", (8, -10))
    for name, color, members in SYSTEMS:
        xs, ys = summary.loc[[c for c, _ in members], "dwer"], summary.loc[[c for c, _ in members], f"{a}_eer"]
        ax.plot(xs, ys, color=color, lw=2, label=name, zorder=2)
        for (c, label), x, y in zip(members, xs, ys):
            off = {"mcadams_0.9": (8, 4), "seedvc_cfg1.0": (-8, 0), "seedvc_cfg0.7": (6, -11)}.get(c, (8, 0))
            point(ax, x, y, None, yerr(c, a), color, label, off)
    ax.set(facecolor="#fcfcfb", xlim=(-0.6, 8.4), ylim=(-4, 56), xlabel="ΔWER (percentage points)",
           title=f"Attacker: {NAMES[a]}")
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
axes[0].set_ylabel("EER (%)  ↑ more privacy")
axes[0].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(RES / "figures" / "fig1_privacy_utility.png", dpi=200)

# Figure 2: attacker disagreement
fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor="#fcfcfb")
ax.plot([0, 55], [0, 55], ls="--", lw=1, color=MUTED)
ax.text(2, 52, "WavLM-SV reports\nmore privacy", color=MUTED, fontsize=8, ha="left", va="top")
ax.text(52, 30, "ECAPA reports\nmore privacy", color=MUTED, fontsize=8, ha="right", va="bottom")


def both(c, color, label, offset):
    r = [ci.loc[(c, a)] for a in ATTACKERS]
    point(ax, r[0].eer, r[1].eer, [[r[0].eer - r[0].lo], [r[0].hi - r[0].eer]],
          [[r[1].eer - r[1].lo], [r[1].hi - r[1].eer]], color, label, offset)


both("original", INK, "original", (8, -8))
for name, color, members in SYSTEMS:
    ax.plot(summary.loc[[c for c, _ in members], "ecapa_eer"], summary.loc[[c for c, _ in members], "wavlm_eer"],
            color=color, lw=2, label=name, zorder=2)
    for c, label in members:
        if name == "Seed-VC":  # the three settings overlap; one label for the cluster
            both(c, color, "", (0, 0))
        else:
            both(c, color, label, (8, 5) if c == "mcadams_0.9" else (8, -6))
ax.text(47.5, 39, "CFG 0.0 / 0.7 / 1.0", color=INK, fontsize=8, ha="center", va="top")
ax.set(facecolor="#fcfcfb", xlim=(-2, 55), ylim=(-2, 55), aspect="equal",
       xlabel=f"{NAMES['ecapa']} EER (%)", ylabel=f"{NAMES['wavlm']} EER (%)")
ax.grid(color="#e1e0d9", lw=0.8)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(RES / "figures" / "fig2_attacker_disagreement.png", dpi=200)

# Figure 6 (M7): attacker knowledge. ignorant -> lazy-informed -> lazy-informed + adapted backend
if lazy is not None:
    anon = [c for _, _, members in SYSTEMS for c, _ in members]
    stages = [("", "o", True, "ignorant (original enrollment)"),
              ("_lazy", "o", False, "lazy-informed (anonymized enrollment)")]
    if informed is not None:
        stages.append(("_backend", "s", False, "lazy-informed + backend trained on own anonymized speech"))
    for (s0, *_), (s1, *_) in zip(stages, stages[1:]):
        print(f"\n{s1[1:]} minus {s0[1:] or 'ignorant'} EER (paired speaker bootstrap, 95 % CI):")
        for c in anon:
            cells = []
            for a in ATTACKERS:
                d = boot[c + s1, a] - boot[c + s0, a]
                cells.append(f"{a} {points.loc[c + s0, f'{a}_eer']:5.2f} -> {points.loc[c + s1, f'{a}_eer']:5.2f} "
                             f"({np.percentile(d, 2.5):+.2f} to {np.percentile(d, 97.5):+.2f})")
            print(f"  {c:14s} " + "   ".join(cells))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, facecolor="#fcfcfb")
    offsets = np.linspace(-0.22, 0.22, len(stages))
    for ax, a in zip(axes, ATTACKERS):
        ax.axhline(50, ls="--", lw=1, color=MUTED)
        for i, c in enumerate(anon):
            color = SYSTEMS[0][1] if c.startswith("mcadams") else SYSTEMS[1][1]
            rs = [ci.loc[(c + s, a)] for s, *_ in stages]
            ax.plot(i + offsets, [r.eer for r in rs], color=MUTED, lw=1, zorder=1)
            for dx, r, (_, marker, filled, _) in zip(offsets, rs, stages):
                ax.errorbar(i + dx, r.eer, yerr=[[r.eer - r.lo], [r.hi - r.eer]], fmt=marker, ms=6, color=color,
                            mfc=color if filled else "#fcfcfb", mec=color, mew=1.5, elinewidth=1, capsize=0, zorder=3)
        ax.set_xticks(range(len(anon)), [label for _, _, members in SYSTEMS for _, label in members], fontsize=8)
        ax.set(facecolor="#fcfcfb", ylim=(-2, 56), title=f"Attacker: {NAMES[a]}")
        ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    axes[0].set_ylabel("EER (%)  ↑ more privacy")
    handles = [plt.Line2D([], [], color=SYSTEMS[0][1], lw=0, marker="o", label="McAdams"),
               plt.Line2D([], [], color=SYSTEMS[1][1], lw=0, marker="o", label="Seed-VC")]
    handles += [plt.Line2D([], [], color=INK, lw=0, marker=m, mfc=INK if f else "#fcfcfb", label=label)
                for _, m, f, label in stages]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(RES / "figures" / "fig6_lazy_informed.png", dpi=200)
