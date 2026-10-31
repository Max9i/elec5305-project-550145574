% Anonymize every utterance of a list with McAdams, for each alpha.
% Writes data/anon/mcadams_<alpha>/<utt>.wav, and results/timing_mcadams.csv for the trial list.
%   matlab -batch "cd matlab; run_mcadams"                      % trials (default)
%   matlab -batch "cd matlab; list = 'enroll'; run_mcadams"     % enrollment, for the lazy-informed attacker
%   matlab -batch "cd matlab; list = 'backend'; run_mcadams"    % dev-clean, for the informed backend
if ~exist('list', 'var'), list = 'trials'; end
root = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(root, 'data', 'lists', [list '.csv']), 'TextType', 'string');
if ~ismember('subset', T.Properties.VariableNames), T.subset = repmat("test-clean", height(T), 1); end
alphas = [0.9 0.8 0.7 0.6];
condition = compose("mcadams_%.1f", alphas');
[audio_s, proc_s] = deal(zeros(numel(alphas), 1));
for i = 1:numel(alphas)
    out = fullfile(root, 'data', 'anon', condition(i));
    [~, ~] = mkdir(out);
    for k = 1:height(T)
        u = T.utt(k);
        id = split(u, '-');
        [x, fs] = audioread(fullfile(root, 'data', 'raw', 'LibriSpeech', T.subset(k), id(1), id(2), u + ".flac"));
        t = tic;
        y = mcadams_anon(x, fs, alphas(i));
        proc_s(i) = proc_s(i) + toc(t);
        audio_s(i) = audio_s(i) + numel(x) / fs;
        audiowrite(fullfile(out, u + ".wav"), y, fs);
    end
    fprintf('%s: %.0f s of audio in %.1f s\n', condition(i), audio_s(i), proc_s(i));
end
if strcmp(list, 'trials')
    rtf = proc_s ./ audio_s;  % processing time per second of audio
    writetable(table(condition, audio_s, proc_s, rtf), fullfile(root, 'results', 'timing_mcadams.csv'));
end
