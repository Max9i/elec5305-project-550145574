"""Self-check for acoustics.compare():  python python/test_acoustics.py"""
import librosa

from acoustics import LIBRI, SR, compare

x = librosa.load(LIBRI / "1089/134686/1089-134686-0000.flac", sr=SR)[0]
shift, corr, kept, lsd = compare(x, x)
assert shift == 0 and corr > 0.999 and kept == 1 and lsd < 0.01, (shift, corr, kept, lsd)
shift, corr, kept, lsd = compare(x, librosa.effects.pitch_shift(x, sr=SR, n_steps=2))
assert abs(shift - 2) < 0.3 and corr > 0.9 and kept > 0.8 and lsd > 1, (shift, corr, kept, lsd)  # +2 st, same intonation
print(f"acoustics ok (pitch shift +2 st measured as {shift:+.2f} st, corr {corr:.3f}, LSD {lsd:.1f} dB)")
