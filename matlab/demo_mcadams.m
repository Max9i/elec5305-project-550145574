function demo_mcadams(utt)
%DEMO_MCADAMS Example analysis of one utterance (README M2).
%   demo_mcadams("121-127105-0006") writes samples/<utt>_original.wav and
%   samples/<utt>_mcadams_<alpha>.wav, plus results/figures/mcadams_<utt>.png with the waveform and
%   spectrogram (original vs. alpha = 0.8), the LPC poles of the loudest frame before and after
%   warping, and that frame's LPC envelope for every alpha.
root = fileparts(fileparts(mfilename('fullpath')));
utt = string(utt);
id = split(utt, '-');
[x, fs] = audioread(fullfile(root, 'data', 'raw', 'LibriSpeech', 'test-clean', id(1), id(2), utt + ".flac"));
alphas = [0.9 0.8 0.7 0.6];
ramp = ["#86b6ef" "#3987e5" "#1c5cab" "#0d366b"];  % one hue: light = weak, dark = strong warping
ink = "#52514e"; muted = "#898781";

Y = arrayfun(@(a) mcadams_anon(x, fs, a), alphas, 'UniformOutput', false);
audiowrite(fullfile(root, 'samples', utt + "_original.wav"), x, fs);
for i = 1:numel(alphas)
    audiowrite(fullfile(root, 'samples', utt + compose("_mcadams_%.1f.wav", alphas(i))), Y{i}, fs);
end

% LPC model of the loudest 20 ms frame (a vowel)
N = round(0.02 * fs);
[~, c] = max(movsum(x .^ 2, N));
[a, g] = lpc(x(c - N/2 + (1:N)) .* hann(N), 20);

fig = figure('Visible', 'off', 'Position', [0 0 1300 620], 'Color', '#fcfcfb');
tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, utt + ": LPC/McAdams anonymisation");
sig = {x, Y{2}};
name = ["original", "\alpha = 0.8"];
S = cell(1, 2);
for i = 1:2
    [S{i}, f, t] = spectrogram(sig{i}, hann(512), 384, 512, fs);
end
top = 20 * log10(max(cellfun(@(s) max(abs(s(:))), S)));
cmap = interp1([0 0.5 1], hex2rgb(["#fcfcfb"; "#3987e5"; "#0d366b"]), linspace(0, 1, 256));
for i = 1:2
    nexttile(tl, 3*i - 2);
    plot((0:numel(x) - 1) / fs, sig{i}, 'Color', ink);
    axis tight; ylim([-1 1] * max(abs(x)));
    title(name(i) + ": waveform"); xlabel('time (s)');
    nexttile(tl, 3*i - 1);
    imagesc(t, f / 1000, 20 * log10(abs(S{i}) + eps)); axis xy;
    clim([top - 80, top]); colormap(gca, cmap);
    title(name(i) + ": spectrogram"); xlabel('time (s)'); ylabel('frequency (kHz)');
end

nexttile(tl, 3);
r = roots(a); rw = roots(mcadams_warp(a, 0.8));
th = linspace(0, pi, 200);
plot(cos(th), sin(th), 'Color', "#e1e0d9", 'HandleVisibility', 'off'); hold on
fixed = sprintf('\\phi = 1 rad = %.2f kHz', fs / (2*pi) / 1000);  % fixed point of phi^alpha
plot([0 cos(1)], [0 sin(1)], ':', 'Color', muted, 'DisplayName', fixed);
plot(real(r), imag(r), 'o', 'Color', ink, 'MarkerSize', 7, 'DisplayName', 'original');
plot(real(rw), imag(rw), 'x', 'Color', ramp(2), 'MarkerSize', 9, 'LineWidth', 1.5, 'DisplayName', '\alpha = 0.8');
axis equal; xlim([-1.05 1.05]); ylim([-0.05 1.05]);
legend('Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
title('poles of 1/A(z), loudest frame');

nexttile(tl, 6);
[h, fz] = freqz(sqrt(g), a, 512, fs);
plot(fz / 1000, 20 * log10(abs(h)), 'Color', ink, 'LineWidth', 2); hold on
for i = 1:numel(alphas)
    plot(fz / 1000, 20 * log10(abs(freqz(sqrt(g), mcadams_warp(a, alphas(i)), 512, fs))), ...
        'Color', ramp(i), 'LineWidth', 1.5);
end
xline(fs / (2*pi) / 1000, ':', fixed, 'Color', muted, 'LabelVerticalAlignment', 'bottom');
legend(["original", compose("\\alpha = %.1f", alphas)], 'Location', 'southwest', 'Box', 'off');
xlabel('frequency (kHz)'); ylabel('dB'); title('LPC envelope, loudest frame');

set(findobj(fig, 'Type', 'axes'), 'XColor', muted, 'YColor', muted, 'Box', 'off');
exportgraphics(fig, fullfile(root, 'results', 'figures', "mcadams_" + utt + ".png"), 'Resolution', 150);
end
