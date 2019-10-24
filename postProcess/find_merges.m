function rez = find_merges(rez, flag)
% this function merges clusters based on template correlation
% however, a merge is veto-ed if refractory period violations are introduced

ops = rez.ops;
dt = 1/1000;

Xsim = rez.simScore; % this is the pairwise similarity score
Nk = size(Xsim,1);
Xsim = Xsim - diag(diag(Xsim)); % remove the diagonal of ones


% note on columns in st3, this is copied from splitAllClusters
% - templates correspond to features defined in W, clusters correspond to spike classifications that may include 1 or more templates
% - original templates live in column 2, this will also be the original cluster column
% - column 6 will be used as the modifiable template column, column 7 will be used as the modifiable cluster column
% - both split and merge will modify the current "cluster" column
% - split will also modify the current template column, but merge will not 

if rez.st3_template_col == 2
    % need to create column 6
    rez.st3(:, 6) = rez.st3(:, 2);
    rez.st3_template_col = 6;
end
if rez.st3_cluster_col == 2
    rez.st3(:, 7) = rez.st3(:, rez.st3_template_col);
    rez.st3_cluster_col = 7;
end

% we'll modify only the cluster column
cluster_col = rez.st3_cluster_col;

if ~isfield(rez, 'mergecount')
    rez.mergecount = ones(Nk, 1);
end
if ~isfield(rez, 'mergedst')
    rez.mergedst = nan(Nk, 1);
end

% sort by firing rate first
nspk = accumarray(rez.st3(:, cluster_col), 1);
[~, isort] = sort(nspk);

if ~flag
  % if the flag is off, then no merges are performed
  % this function is then just used to compute cross- and auto- correlograms
   rez.R_CCG = Inf * ones(Nk);
   rez.Q_CCG = Inf * ones(Nk);
   rez.K_CCG = {};
end

nmerge = 0;
prog = ProgressBar(Nk, 'Searching for cluster auto-merges');
for j = 1:Nk
    prog.update(j);
    s1 = rez.st3(rez.st3(:,cluster_col)==isort(j), 1)/ops.fs; % find all spikes from this cluster
    if numel(s1)~=nspk(isort(j))
        prog.pause_for_output();
        warning('Lost track of spike counts on cluster %d', isort(j)) %this is a check for myself to make sure new cluster are combined correctly into bigger clusters
    end

    % sort all the pairs of this neuron, discarding any that have fewer spikes
    [ccsort, ix] = sort(Xsim(isort(j),:) .* (nspk'>numel(s1)), 'descend');
    ienu = find(ccsort<.5, 1) - 1; % find the first pair which has too low of a correlation

    % for all pairs above 0.5 correlation
    for k = 1:ienu
        s2 = rez.st3(rez.st3(:, cluster_col)==ix(k), 1)/ops.fs; % find the spikes of the pair
        % compute cross-correlograms, refractoriness scores (Qi and rir), and normalization for these scores
        [K, Qi, Q00, Q01, rir] = ccg(s1, s2, 500, dt);
        Q = min(Qi/(max(Q00, Q01))); % normalize the central cross-correlogram bin by its shoulders OR by its mean firing rate
        R = min(rir); % R is the estimated probability that any of the center bins are refractory, and kicks in when there are very few spikes

        if flag
            if Q<.2 && R<.05 % if both refractory criteria are met
                i = ix(k);
                % now merge isort(j) into i and move on
                rez.st3(rez.st3(:, cluster_col)==isort(j), cluster_col) = i; % simply overwrite all the spikes of cluster j with i (i>j by construction)
                nspk(i) = nspk(i) + nspk(isort(j)); % update number of spikes for cluster i
%                 fprintf('merged %d into %d \n', isort(j), i)
                rez.mergecount(i) = rez.mergecount(i) + rez.mergecount(isort(j));
                rez.mergecount(isort(j)) = 0;
                rez.mergedst(isort(j)) = i;
%                 if isfield(rez, 'splitdst') && isfield(rez, 'splitsrc')
%                     rez.splitdst(rez.splitdst == isort(j)) = i;
%                     if rez.splitsrc(i) == isort(j)
%                         % we're re-merging a split
%                         rez.splitsrc(i) = NaN;
%                     end
%                     rez.splitsrc(rez.splitsrc == isort(j)) = rez.splitsrc(i);
%                     rez.splitsrc(isort(j)) = NaN;
%                     rez.splitdst(isort(j)) = NaN;
%                 end

                % YOU REALLY SHOULD MAKE SURE THE PC CHANNELS MATCH HERE
                nmerge = nmerge + 1;
                break; % if a pair is found, we don't need to keep going (we'll revisit this cluster when we get to the merged cluster)
            end
        else
          % sometimes we just want to get the refractory scores and CCG
            rez.R_CCG(isort(j), ix(k)) = R;
            rez.Q_CCG(isort(j), ix(k)) = Q;

            rez.K_CCG{isort(j), ix(k)} = K;
            rez.K_CCG{ix(k), isort(j)} = K(end:-1:1); % the CCG is "antisymmetrical"
        end
    end
end
prog.finish();

if ~flag
    rez.R_CCG  = min(rez.R_CCG , rez.R_CCG'); % symmetrize the scores
    rez.Q_CCG  = min(rez.Q_CCG , rez.Q_CCG');
end

fprintf('Merged %d cluster pairs\n', nmerge);
