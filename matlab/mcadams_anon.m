function y = mcadams_anon(x, fs, alpha)
%MCADAMS_ANON Speaker anonymisation with the McAdams coefficient (Patino et al., Interspeech 2021).
%   y = mcadams_anon(x, fs, alpha) cuts mono speech x into 20 ms frames with a 10 ms hop, fits an
%   order-20 LPC model A(z) to each frame, keeps the residual e[n] = A(z)s[n] (excitation, so F0
%   and timing), warps the pole angles of 1/A(z) with mcadams_warp and resynthesises e[n]/A'(z)
%   with weighted overlap-add. alpha = 1 returns x up to rounding error.
H = round(0.010 * fs); N = 2 * H; p = 20;
w = sqrt(hann(N, 'periodic'));                 % analysis = synthesis window: w.^2 overlap-adds to 1
xp = [zeros(H, 1); x(:); zeros(N, 1)];         % pad so every input sample lies in two frames
idx = (1:H:numel(xp) - N + 1) + (0:N - 1)';    % N x frames sample indices
F = xp(idx) .* w;                              % windowed frames, one per column
A = lpc(F, p);                                 % one row of LPC coefficients per frame
yp = zeros(size(xp));
for k = 1:size(F, 2)
    if any(isnan(A(k, :))), continue, end      % all-zero frame: no LPC model, output stays zero
    e = filter(A(k, :), 1, F(:, k));
    yp(idx(:, k)) = yp(idx(:, k)) + w .* filter(1, mcadams_warp(A(k, :), alpha), e);
end
y = yp(H + 1:H + numel(x));
y = y * (max(abs(x)) / max(abs(y)));           % keep the input peak level (audiowrite clips at +-1)
end
