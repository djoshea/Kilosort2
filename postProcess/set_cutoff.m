function rez = set_cutoff(rez)

ops = rez.ops;
dt = 1/1000;
markSplitsOnly = getOr(rez.ops, 'markSpitsOnly', false);

if markSplitsOnly
    cluster_col = 6; % use post-merge, pre-split cluster assignments
else
    cluster_col = 7; % use post split cluster assignments
end
Nk = max(rez.st3(:, cluster_col)); 

spike_valid = true(size(rez.st3, 1), 1);

% sort by firing rate first
rez.good = zeros(Nk, 1);
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
%     rez.st3(ix(vexp<=Th), 6) = 0;
    
    if rem(j,100)==1
%        fprintf('%d \n', j) 
    end
end

% we sometimes get NaNs, why?
rez.est_contam_rate(isnan(rez.est_contam_rate)) = 1;

% hold onto the invalid spikes in .st3_cutoff_invalid before removing them
% ix = rez.st3(:, 6)==0;
ix = ~spike_valid;
rez.st3_cutoff_invalid = rez.st3(ix, :);
rez.cProj_cutoff_invalid = rez.cProj(ix, :);
rez.cProjPC_cutoff_invalid = rez.cProjPC(ix, :, :);

% now delete them
rez.st3(ix, :) = [];
if ~isempty(rez.cProj)
    rez.cProj(ix, :) = [];
    rez.cProjPC(ix, :,:) = [];
end
