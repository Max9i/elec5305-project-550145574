% Anonymise every trial utterance in data/lists/trials.csv with McAdams, for each alpha.
% Writes data/anon/mcadams_<alpha>/<utt>.wav and results/timing_mcadams.csv.
%   matlab -batch "cd matlab; run_mcadams"
root = fileparts(fileparts(mfilename('fullpath')));
T = readtable(fullfile(root, 'data', 'lists', 'trials.csv'), 'TextType', 'string');
alphas = [0.9 0.8 0.7 0.6];
condition = compose("mcadams_%.1f", alphas');
[audio_s, proc_s] = deal(zeros(numel(alphas), 1));
for i = 1:numel(alphas)
    out = fullfile(root, 'data', 'anon', condition(i));
    [~, ~] = mkdir(out);
    for u = T.utt'
        id = split(u, '-');
        [x, fs] = audioread(fullfile(root, 'data', 'raw', 'LibriSpeech', 'test-clean', id(1), id(2), u + ".flac"));
        t = tic;
        y = mcadams_anon(x, fs, alphas(i));
        proc_s(i) = proc_s(i) + toc(t);
        audio_s(i) = audio_s(i) + numel(x) / fs;
        audiowrite(fullfile(out, u + ".wav"), y, fs);
    end
    fprintf('%s: %.0f s of audio in %.1f s\n', condition(i), audio_s(i), proc_s(i));
end
rtf = proc_s ./ audio_s;  % processing time per second of audio
writetable(table(condition, audio_s, proc_s, rtf), fullfile(root, 'results', 'timing_mcadams.csv'));
