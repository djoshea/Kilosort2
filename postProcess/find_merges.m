function rez = find_merges(rez, flag)

ops = rez.ops;
dt = 1/1000;

Xsim = rez.simScore;
Nk = size(Xsim,1);
Xsim = Xsim - diag(diag(Xsim));

template_col = rez.st3_template_col;
if isfield(rez, 'st3_merge_col')
    % no need to initialize, already set
    cluster_col = rez.st3_merge_col;
else
    % merge hasn't been called yet
    cluster_col = size(rez.st3, 2) + 1;
    % initialze from current template_col
    rez.st3(:, cluster_col) = rez.st3(:, template_col);
end
rez.st3_merge_col = cluster_col;
rez.st3_cluster_col = cluster_col;

if ~isfield(rez, 'mergecount')
    rez.mergecount = ones(Nk, 1);
end
if ~isfield(rez, 'mergedst')
    rez.mergedst = nan(Nk, 1);
end

% sort by firing rate first
% nspk = zeros(Nk, 1);
% for j = 1:Nk
%     nspk(j) = sum(rez.st3(:,2)==j);        
% end
nspk = accumarray(rez.st3(:, cluster_col), 1); 
[~, isort] = sort(nspk);

if ~flag
   rez.R_CCG = Inf * ones(Nk);
   rez.Q_CCG = Inf * ones(Nk);
   rez.K_CCG = {};
end

nmerge = 0;
prog = ProgressBar(Nk, 'Searching for cluster auto-merges');
for j = 1:Nk
    prog.update(j);
    s1 = rez.st3(rez.st3(:,cluster_col)==isort(j), 1)/ops.fs;
    if numel(s1)~=nspk(isort(j))
        prog.pause_for_output();
        warning('Lost track of spike counts on cluster %d', isort(j))
    end    
    [ccsort, ix] = sort(Xsim(isort(j),:) .* (nspk'>numel(s1)), 'descend');
    ienu = find(ccsort<.5, 1) - 1;
    
    for k = 1:ienu
        s2 = rez.st3(rez.st3(:, cluster_col)==ix(k), 1)/ops.fs;
        [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
        Q = min(Qi/(max(Q00, Q01)));
%         Q = min(Qi/Q01);
        R = min(rir);
        
        if flag
            if Q<.2 && R<.05
                i = ix(k);
                % now merge isort(j) into i and move on
                rez.st3(rez.st3(:, cluster_col)==isort(j), cluster_col) = i;
                nspk(i) = nspk(i) + nspk(isort(j));
%                 fprintf('merged %d into %d \n', isort(j), i)
                rez.mergecount(i) = rez.mergecount(i) + rez.mergecount(isort(j));
                rez.mergecount(isort(j)) = 0;
                rez.mergedst(isort(j)) = i;
                if isfield(rez, 'splitdst') && isfield(rez, 'splitsrc')
                    rez.splitdst(rez.splitdst == isort(j)) = i;
                    if rez.splitsrc(i) == isort(j)
                        % we're re-merging a split
                        rez.splitsrc(i) = NaN;
                    end
                    rez.splitsrc(rez.splitsrc == isort(j)) = rez.splitsrc(i);
                    rez.splitsrc(isort(j)) = NaN;
                    rez.splitdst(isort(j)) = NaN;
                end
                
                % YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                nmerge = nmerge + 1;
                break;
            end
        else
            rez.R_CCG(isort(j), ix(k)) = R;
            rez.Q_CCG(isort(j), ix(k)) = Q;
            
            rez.K_CCG{isort(j), ix(k)} = K;                        
            rez.K_CCG{ix(k), isort(j)} = K;
        end
    end   
end
prog.finish();

if ~flag
    rez.R_CCG  = min(rez.R_CCG , rez.R_CCG');
    rez.Q_CCG  = min(rez.Q_CCG , rez.Q_CCG');
end

fprintf('Merged %d cluster pairs\n', nmerge);


