function rez = set_cutoff(rez)

ops = rez.ops;
dt = 1/1000;

if size(rez.st3, 2) == 7
    cluster_col = 7; % use post split cluster assignments
elseif size(rez.st3, 2) == 6
    cluster_col = 6;
else
    cluster_col = 2;
end
Nk = max(rez.st3(:, cluster_col)); 

spike_valid = true(size(rez.st3, 1), 1);

% sort by firing rate first
rez.good = zeros(Nk, 1);
prog = ProgressBar(Nk, 'Setting cluster specific spike cutoffs');
for j = 1:Nk
    ix = find(rez.st3(:,cluster_col)==j);        
    ss = rez.st3(ix,1)/ops.fs;
    if numel(ss)==0
        continue;
    end
    
    vexp = rez.st3(ix,4);
    
    Th = ops.Th(1);    
    

    fcontamination = 0.1; % acceptable contamination rate

    rez.est_contam_rate(j) = 1;
    while Th>=ops.Th(2)
        st = ss(vexp>Th);
        if isempty(st)
            Th = Th - .5;
            continue;
        end
        [K, Qi, Q00, Q01, rir] = ccg(st, st, 500, dt);
        Q = min(Qi/(max(Q00, Q01)));
        R = min(rir);
        if Q>fcontamination || R>.05                
           break; 
        else
            if Th==ops.Th(1) && Q<.05
                fcontamination = min(.05, max(.01, Q*2));
            end
            rez.good(j) = 1;
            rez.est_contam_rate(j) = Q;
            Th = Th - .5;
        end        
    end
    Th = Th + .5;
    
    rez.Ths(j) = Th;
    % just mark it valid, we'll take care of clearing it later
    spike_valid(ix(vexp<=Th)) = false;
%     rez.st3(ix(vexp<=Th), 6) = 0; % not used anymore, we want to slice it off into st3_cutoff_invalid
    
    prog.update(j);
end
prog.finish();

% we sometimes get NaNs, why?
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
    rez.cProj(ix, :) = [];
end
if ~isempty(rez.cProjPC)
    rez.cProjPC(ix, :,:) = [];
end
