function rez = set_cutoff(rez)
% after everything else is done, this function takes spike trains and cuts off
% any noise they might have picked up at low amplitude values
% We look for bimodality in the amplitude plot, thus setting an individual threshold
% for each neuron.
% Also, this function calls "good" and "bad" clusters based on the auto-correlogram

ops = rez.ops;
dt = 1/1000; % step size for CCG binning

if size(rez.st3, 2) == 7
    cluster_col = 7; % use post merge cluster assignments
elseif size(rez.st3, 2) == 6
    cluster_col = 6; % use post split cluster assignments
else
    cluster_col = 2;
end
Nk = max(rez.st3(:, cluster_col));  % number of templates

spike_valid = true(size(rez.st3, 1), 1);

% sort by firing rate first
rez.good = zeros(Nk, 1);
prog = ProgressBar(Nk, 'Setting cluster specific spike cutoffs');
for j = 1:Nk
    ix = find(rez.st3(:,cluster_col)==j); % find all spikes from this neuron
    ss = rez.st3(ix,1)/ops.fs; % convert to seconds
    if numel(ss)==0
        continue; % break if there are no spikes
    end

    vexp = rez.st3(ix,4); % vexp is the relative residual variance of the spikes

    Th = ops.Th(1); % start with a high threshold

    fcontamination = 0.1; % acceptable contamination rate

    rez.est_contam_rate(j) = 1;
    while Th>=ops.Th(2)
      % continually lower the threshold, while the estimated unit contamination is low
        st = ss(vexp>Th); % take spikes above the current threshold
        if isempty(st)
            Th = Th - .5; % if there are no spikes, we need to keep lowering the threshold
            continue;
        end
        [K, Qi, Q00, Q01, rir] = ccg(st, st, 500, dt); % % compute the auto-correlogram with 500 bins at 1ms bins
        Q = min(Qi/(max(Q00, Q01))); % this is a measure of refractoriness
        R = min(rir); % this is a second measure of refractoriness (kicks in for very low firing rates)
        if Q>fcontamination || R>.05 % if the unit is already contaminated, we break, and use the next higher threshold
           break;
        else
            if Th==ops.Th(1) && Q<.05
              % only on the first iteration, we consider if the unit starts well isolated
              % if it does, then we put much stricter criteria for isolation
              % to make sure we don't settle for a relatively high contamination unit
                fcontamination = min(.05, max(.01, Q*2));

                % if the unit starts out contaminated, we will settle with the higher contamination rate
            end
            rez.good(j) = 1; % this unit is good, because we will stop lowering the threshold when it becomes bad
            Th = Th - .5; % try the next lower threshold
        end
    end
    Th = Th + .5;  % we exited the loop because the contamination was too high. We revert to the higher threshold

    % just mark it valid, we'll take care of clearing it later
    spike_valid(ix(vexp<=Th)) = false; % valid spikes are above the current threshold
    st = ss(vexp>Th); % take spikes above the current threshold
    [K, Qi, Q00, Q01, rir] = ccg(st, st, 500, dt); % % compute the auto-correlogram with 500 bins at 1ms bins
    Q = min(Qi/(max(Q00, Q01))); % this is a measure of refractoriness
    rez.est_contam_rate(j) = Q; % this score will be displayed in Phy
    rez.Ths(j) = Th; % store the threshold for potential debugging

    prog.update(j);
end
prog.finish();

% we sometimes get NaNs, why? replace with full contamination
rez.est_contam_rate(isnan(rez.est_contam_rate)) = 1;

% hold onto the invalid spikes in .st3_cutoff_invalid before removing them
% ix = rez.st3(:, 6)==0;
fprintf('Invalidating %d / %d spikes (%.2f %%)\n', nnz(~spike_valid), numel(spike_valid), 100*nnz(~spike_valid) / numel(spike_valid));
ix = ~spike_valid;

if isfield(rez, 'st3_cutoff_invalid')
    nColsCurrent = size(rez.st3_cutoff_invalid, 2);
    nColsNeeded = size(rez.st3, 2);
    if nColsCurrent < nColsNeeded
        rez.st3_cutoff_invalid = cat(2, rez.st3_cutoff_invalid, zeros(size(rez.st3_cutoff_invalid, 1), nColsNeeded - nColsCurrent));
        % copy over template column to clusters so that we don't depend on the order that merging and splitting is performed
        if nColsCurrent < 6 && nColsNeeded >= 6
            rez.st3_cutoff_invalid(:, 6) = rez.st3_cutoff_invalid(:, 2);
        end
        if nColsCurrent < 7 && nColsNeeded >= 7
            rez.st3_cutoff_invalid(:, 7) = rez.st3_cutoff_invalid(:, 6);
        end
    end
    rez.st3_cutoff_invalid = cat(1, rez.st3_cutoff_invalid, rez.st3(ix, :));
    rez.cProj_cutoff_invalid = cat(1, rez.cProj_cutoff_invalid, rez.cProj(ix, :));
    rez.cProjPC_cutoff_invalid = cat(1, rez.cProjPC_cutoff_invalid, rez.cProjPC(ix, :, :));
else
    rez.st3_cutoff_invalid = rez.st3(ix, :);
    rez.cProj_cutoff_invalid = rez.cProj(ix, :);
    rez.cProjPC_cutoff_invalid = rez.cProjPC(ix, :, :);
end

% now delete them
rez.st3(ix, :) = [];
if ~isempty(rez.cProj)
    rez.cProj(ix, :) = []; % remove their template projections too
end
if ~isempty(rez.cProjPC)
    rez.cProjPC(ix, :,:) = [];  % and their PC projections
end
