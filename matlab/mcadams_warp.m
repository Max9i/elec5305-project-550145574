function aNew = mcadams_warp(a, alpha)
%MCADAMS_WARP McAdams transformation of an LPC polynomial A(z).
%   aNew = mcadams_warp(a, alpha) moves every complex pole r*exp(j*phi) of 1/A(z), 0 < phi < pi,
%   to r*exp(j*phi^alpha) (phi in radians) and mirrors it to the conjugate. Real poles are kept.
%   The radius, and with it the formant bandwidth, is unchanged, so a stable filter stays stable.
%   phi = 1 rad (fs/2pi Hz) is a fixed point: for alpha < 1, formants below it move up and
%   formants above it move down.
r = roots(a);
up = r(imag(r) > 0);                                   % one pole of each conjugate pair
up = abs(up) .* exp(1j * min(angle(up) .^ alpha, pi));
aNew = real(poly([up; conj(up); r(imag(r) == 0)]));
end
