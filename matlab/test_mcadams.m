% Self-check for mcadams_warp and mcadams_anon:  matlab -batch "cd matlab; test_mcadams"
root = fileparts(fileparts(mfilename('fullpath')));
[x, fs] = audioread(fullfile(root, 'data', 'raw', 'LibriSpeech', 'test-clean', '1089', '134686', '1089-134686-0000.flac'));

% warping keeps every pole radius and maps each angle phi to phi^alpha
a = lpc(x(8001:8320) .* hann(320), 20);
r = roots(a); r = r(imag(r) > 0);
rw = roots(mcadams_warp(a, 0.8)); rw = rw(imag(rw) > 0);
assert(max(abs(sort(abs(r)) - sort(abs(rw)))) < 1e-6, 'pole radii changed')
assert(max(abs(sort(angle(r) .^ 0.8) - sort(angle(rw)))) < 1e-6, 'pole angles not phi^alpha')

% alpha = 1 reconstructs the input, alpha = 0.8 changes it
snr = @(y) 10 * log10(sum(x .^ 2) / sum((x - y) .^ 2));
s1 = snr(mcadams_anon(x, fs, 1));
y = mcadams_anon(x, fs, 0.8);
assert(s1 > 60, 'alpha = 1 does not reconstruct the input (SNR %.1f dB)', s1)
assert(numel(y) == numel(x) && all(isfinite(y)) && snr(y) < 20, 'alpha = 0.8 output looks wrong')
fprintf('mcadams ok: alpha = 1 SNR %.1f dB, alpha = 0.8 SNR %.1f dB\n', s1, snr(y))
