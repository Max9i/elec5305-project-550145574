"""M6 figures (README §8): 3 acoustic change vs. privacy, 4 example analysis, 5 speaker-level risk.

    python python/plot_m6.py        # after analyse_m6.py has cached results/acoustics/original_f0.csv
"""
import librosa
import matplotlib
import numpy as np
import pandas as pd

from acoustics import f0
from analyse_m6 import RES, ROOT, far5_z, rho, table, trial_z, trials
from evaluate import NAMES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURFACE = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"         # McAdams, Seed-VC (as in Figures 1 and 2)
SEX = {"F": "#1baf7a", "M": "#4a3aa7"}       # female, male
MCADAMS = [f"mcadams_{a}" for a in ("0.9", "0.8", "0.7", "0.6")]
SEEDVC = [f"seedvc_cfg{c}" for c in ("0.0", "0.7", "1.0")]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "font.size": 9,
                     "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})
summary = pd.read_csv(RES / "summary.csv").set_index("condition")
acoustic = pd.read_csv(RES / "acoustics.csv").set_index("condition")
acoustic.loc["original", "env_lsd_db"] = 0

# Figure 3: (a) EER vs. LPC-envelope change per condition, (b) Seed-VC: WavLM-SV identifiability vs. natural pitch
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 3.9), gridspec_kw={"width_ratios": [1, 1.1]})
for a, style, fill in [("ecapa", "-", None), ("wavlm", "--", SURFACE)]:
    for conds, color in [(["original"] + MCADAMS, BLUE), (SEEDVC, ORANGE)]:
        ax.plot(acoustic.loc[conds, "env_lsd_db"], summary.loc[conds, f"{a}_eer"], style, color=color, lw=1.8,
                marker="o", ms=6, mfc=fill or color, mec=color, mew=1.5)
for c, label in zip(MCADAMS, ["α 0.9", "α 0.8", "α 0.7", "α 0.6"]):
    ax.annotate(label, (acoustic.env_lsd_db[c], summary.ecapa_eer[c]), xytext=(6, -10), textcoords="offset points",
                fontsize=8, color=INK)
ax.annotate("Seed-VC\nCFG 0.0–1.0", (acoustic.env_lsd_db[SEEDVC].max(), 44), xytext=(8, -18),
            textcoords="offset points", fontsize=8, color=INK)
ax.annotate("original", (0, summary.ecapa_eer["original"]), xytext=(6, -10), textcoords="offset points",
            fontsize=8, color=INK)
handles = [plt.Line2D([], [], color=BLUE, lw=1.8, label="McAdams"), plt.Line2D([], [], color=ORANGE, lw=1.8, label="Seed-VC"),
           plt.Line2D([], [], color=INK, ls="-", marker="o", label=NAMES["ecapa"]),
           plt.Line2D([], [], color=INK, ls="--", marker="o", mfc=SURFACE, label=NAMES["wavlm"])]
ax.legend(handles=handles, frameon=False, loc="center left", fontsize=8)
ax.axhline(50, ls=":", lw=1, color=MUTED)
ax.set(xlabel="LPC envelope distance to the original (dB)", ylabel="EER (%)", ylim=(-2, 55),
       title="(a) Privacy vs. spectral-envelope change")
ax.grid(axis="y", color=GRID, lw=0.8)

d = table("seedvc_cfg0.7")
for s, g in d.groupby("sex"):
    bx.scatter(g.f0_hz, g.z_wavlm, s=14, color=SEX[s], alpha=0.75, lw=0,
               label=f"{'female' if s == 'F' else 'male'} (ρ = {rho(g.f0_hz, g.z_wavlm):+.2f})")
out_f0 = d.f0_anon_hz.median()
bx.axvline(out_f0, ls="--", lw=1, color=ORANGE)
bx.text(out_f0 * 1.02, -3.3, f"Seed-VC output\npitch {out_f0:.0f} Hz", color=INK, fontsize=8)
thr = far5_z("seedvc_cfg0.7", "wavlm")
bx.axhline(thr, ls=":", lw=1, color=MUTED)
bx.text(330, thr + 0.08, "FAR 5 %", color=MUTED, fontsize=8, ha="right")
bx.set(xscale="log", xlabel="original median F0 of the trial (Hz, log scale)", ylabel=f"{NAMES['wavlm']} trial z-score",
       title="(b) Seed-VC CFG 0.7: identifiability vs. natural pitch")
bx.set_xticks([80, 100, 125, 160, 200, 250], ["80", "100", "125", "160", "200", "250"])
bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
bx.legend(frameon=False, loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(RES / "figures" / "fig3_acoustic_change_vs_privacy.png", dpi=200)

# Figure 4: one trial in four versions (spectrogram, F0, attacker z-scores, WER)
UTT = "121-127105-0006"
versions = [("original", None), ("mcadams_0.8", "McAdams α 0.8"), ("mcadams_0.6", "McAdams α 0.6"),
            ("seedvc_cfg0.7", "Seed-VC CFG 0.7")]
fig = plt.figure(figsize=(9.5, 7.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9])
spk, chap, _ = UTT.split("-")
contours = {}
for i, (c, title) in enumerate(versions):
    path = (ROOT / "data/raw/LibriSpeech/test-clean" / spk / chap / f"{UTT}.flac" if c == "original"
            else ROOT / "data/anon" / c / f"{UTT}.wav")
    x = librosa.load(path, sr=16000)[0]
    x = 0.05 * x / np.sqrt(np.mean(x ** 2))  # same level for every panel, as the attackers see it
    contours[c] = f0(x)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=512, hop_length=128)), ref=1.0)
    z = trial_z(c).loc[UTT]
    wer = pd.read_csv(RES / "asr" / f"{c}.csv", dtype={"utt": str}).set_index("utt").wer[UTT]
    ax = fig.add_subplot(gs[i // 2, i % 2])
    ax.imshow(S, origin="lower", aspect="auto", cmap="Blues", vmin=S.max() - 70, vmax=S.max(),
              extent=[0, len(x) / 16000, 0, 8])
    ax.set(ylabel="kHz", title=f"{title or 'original'}\nz: ECAPA {z.z_ecapa:.1f}{'*' if z.hit_ecapa else ''}, "
                                f"WavLM-SV {z.z_wavlm:.1f}{'*' if z.hit_wavlm else ''}   ·   WER {100 * wer:.0f} %")
    ax.title.set_fontsize(8.5)
ax = fig.add_subplot(gs[2, :])
for (c, title), color in zip(versions, [INK, BLUE, "#0d366b", ORANGE]):
    t = np.arange(len(contours[c])) * 0.01
    voiced = np.mean(~np.isnan(contours[c]))
    ax.plot(t, contours[c], color=color, lw=1.8, label=f"{title or 'original'} ({voiced:.0%} of frames voiced)")
ax.set(xlabel="time (s)", ylabel="F0 (Hz)", title="F0 (pYIN; gaps = unvoiced)", ylim=(60, 650))
ax.title.set_fontsize(8.5)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
fig.suptitle(f"{UTT} (female). * = identified at FAR 5 %", fontsize=9.5)
fig.tight_layout()
fig.savefig(RES / "figures" / "fig4_example_analysis.png", dpi=200)

# Figure 5: speaker-level risk, ECAPA vs. WavLM-SV, McAdams 0.8 and Seed-VC 0.7
tr = trials()
sex = tr.groupby("spk").sex.first()
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharex=True, sharey=True)
for ax, (c, title) in zip(axes, [("mcadams_0.8", "McAdams α 0.8"), ("seedvc_cfg0.7", "Seed-VC CFG 0.7")]):
    S = trial_z(c).join(tr.spk).groupby("spk")[["z_ecapa", "z_wavlm"]].mean()
    te, tw = far5_z(c, "ecapa"), far5_z(c, "wavlm")
    ax.axvline(te, ls=":", lw=1, color=MUTED)
    ax.axhline(tw, ls=":", lw=1, color=MUTED)
    for s in ("F", "M"):
        g = S[sex[S.index] == s]
        ax.scatter(g.z_ecapa, g.z_wavlm, s=28, color=SEX[s], lw=0.8, edgecolor=SURFACE,
                   label="female" if s == "F" else "male")
    n = lambda e, w: ((S.z_ecapa > te) == e).__and__((S.z_wavlm > tw) == w).sum()
    for (e, w), (x, y, ha) in {(True, True): (0.98, 0.97, "right"), (True, False): (0.98, 0.03, "right"),
                               (False, True): (0.02, 0.97, "left"), (False, False): (0.02, 0.03, "left")}.items():
        k = n(e, w)
        ax.text(x, y, f"{k} speaker{'' if k == 1 else 's'}", transform=ax.transAxes, ha=ha,
                va="top" if y > 0.5 else "bottom", fontsize=8, color=INK)
    ax.set(title=f"{title}: ρ = {rho(S.z_ecapa, S.z_wavlm):+.2f}", xlabel=f"{NAMES['ecapa']} speaker mean z",
           ylim=(-2.5, 2.7))
    ax.grid(color=GRID, lw=0.8)
axes[0].set_ylabel(f"{NAMES['wavlm']} speaker mean z")
axes[0].legend(frameon=False, loc="center left", fontsize=8)
fig.text(0.5, 0.005, "Dotted lines: identification threshold (FAR 5 %). Upper right = identified by both, "
         "lower right = only by ECAPA-TDNN, lower left = by neither.", ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(RES / "figures" / "fig5_speaker_risk.png", dpi=200)
