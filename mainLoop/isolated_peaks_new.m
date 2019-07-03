function [row, col, mu] = isolated_peaks_new(S1, ops, distrust_samples)

loc_range = getOr(ops, 'loc_range', [5 4]);
long_range = getOr(ops, 'long_range', [30 6]);
Th = ops.spkTh;
nt0 = ops.nt0;

% loc_range = [3  1];
% long_range = [30  6];
if getOr(ops, 'spikeThreshBothDirs', false)
    smin = my_min(S1, loc_range, [1 2]);
    smax = my_min(-S1, loc_range, [1 2]);
    peaks = single((S1<smin+1e-3 & S1<Th) | (-S1<smax+1e-3 & S1>-Th));
else
    % only threshold negative crossings
    smin = my_min(S1, loc_range, [1 2]);
    peaks = single(S1<smin+1e-3 & S1<Th);
end

sum_peaks = my_sum(peaks, long_range, [1 2]);
peaks = peaks .* (sum_peaks<1.2) .* S1;

% exclude temporal buffers
peaks([1:nt0 end-nt0:end], :) = 0;

if nargin > 2 && ~isempty(distrust_samples)
    % we need to remove spikes that would have some overlap with the distrusted samples
    spike_window = -ops.nt0min + [1, ops.nt0];
    
    % build a structuring element with an odd number of elements, with the center element corresponding to the 
    % current sample. then put spike_window(1) ones to the left and spike_window(2) ones to the right
    se = zeros(1 + 2*max(spike_window), 1);
    mid = max(spike_window) + 1;
    start_ones = mid + spike_window(1);
    stop_ones = mid + spike_window(2);
    se(start_ones:stop_ones) = 1;
    distrust_samples = imdilate(distrust_samples, se);
    
    peaks(~distrust_samples, :) = 0;
end

% exclude edge channels 
% noff = 8;
% peaks(:, [1:noff end-noff+ [1:noff]]) = 0;

[row, col, mu] = find(peaks);

mu = - mu;
