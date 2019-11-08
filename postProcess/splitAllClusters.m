function rez = splitAllClusters(rez, flag)
% i call this algorithm "bimodal pursuit"
% split clusters if they have bimodal projections
% the strategy is to maximize a bimodality score and find a single vector projection
% that maximizes it. If the distribution along that maximal projection crosses a
% bimodality threshold, then the cluster is split along that direction
% it only uses the PC features for each spike, stored in rez.cProjPC
%
% if markSplitsOnly is false, copies original template column 2 to 6 (if not present already) and splits template assignments in column st3_template_col
ops = rez.ops;
markSplitsOnly = getOr(ops, 'markSplitsOnly', false);

reproducible = getOr(ops, 'reproducible', false);
roundpow2 = 17;

if ~isfield(rez, 'W_preSplit')
    rez.dWU_preSplit = rez.dWU;
    rez.W_preSplit = rez.W;
    rez.U_preSplit = rez.U;
    rez.mu_preSplit = rez.mu;
end

wPCA = gather(ops.wPCA);

ccsplit = rez.ops.AUCsplit; % this is the threshold for splits, and is one of the main parameters users can change

NchanNear   = min(ops.Nchan, 32);
Nnearest    = min(ops.Nchan, 32);
sigmaMask   = ops.sigmaMask;


% note on columns in st3:
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

% we'll modify both columns
template_col = rez.st3_template_col;
cluster_col = rez.st3_cluster_col;

ik = 0;
Nfilt = size(rez.W,2);
nsplits= 0;

% determine what channels each template lives on
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear); %#ok<ASGLU> % iC is 32 x nCh listing the 32 closest channels to each other channel

ops.nt0min = getOr(ops, 'nt0min', 20); % the waveforms must be aligned to this sample

 % find the peak abs channel for each template
[~, iW] = max(abs(rez.dWU(ops.nt0min, :, :)), [], 2); % iW indicates for each template, on which channel dWU (the average of the spikes) is largest
iW = squeeze(int32(iW));

isplit = (1:Nfilt)'; % keep track of original cluster for each cluster. starts with all clusters being their own origin.
dt = 1/1000;
nccg = 0;

if isfield(rez, 'split_candidate')
    split_candidate = rez.split_candidate;
else
    split_candidate = false(Nfilt, 1);
end
if isfield(rez, 'splitsrc')
    splitsrc = rez.splitsrc;
else
    splitsrc = nan(Nfilt, 1);
end
if isfield(rez, 'splitdst')
    splitdst = rez.splitdst;
else
    splitdst = cell(Nfilt, 1);
end
if isfield(rez, 'splitauc')
    splitauc = rez.splitauc;
else
    splitauc = cell(Nfilt, 1);
end
if isfield(rez, 'split_orig_template')
    split_orig_template = rez.split_orig_template;
else
    split_orig_template = (1:Nfilt)';
end
if isfield(rez, 'splitProjections')
    splitProj = rez.splitProjections;
else
    splitProj = cell(Nfilt, 1);
end

prog = ProgressBar(Nfilt, 'Searching for clusters to split');
while ik<Nfilt
    if rem(ik, 100)==1
        % periodically write updates
        prog.pause_for_output();
        fprintf('Made %d splits, checked %d/%d clusters, nccg %d \n', nsplits, ik, Nfilt, nccg)
    end
    ik = ik+1;
    prog.update(ik);

    isp = find(rez.st3(:,cluster_col)==ik); % get all spikes from this cluster
    nSpikes = numel(isp);
    if  nSpikes<300
       continue; % do not split if fewer than 300 spikes (we cannot estimate cross-correlograms accurately)
    end

    ss = rez.st3(isp,1)/ops.fs; % convert to seconds

    clp0 = rez.cProjPC(isp, :, :); % get the PC projections for these spikes
    %clp0 = gpuArray(clp0(:,:));
    clp0 = clp0(:,:);
    clp = clp0 - mean(clp0,1); % mean center them

    clp = clp - my_conv2(clp, 250, 1); % subtract a running average, because the projections are NOT drift corrected

    % now use two different ways to initialize the bimodal direction
    % the main script calls this function twice, and does both initializations
    if flag
        [u, s, v] = svdecon(clp'); %#ok<ASGLU>
        w = u(:,1); % initialize with the top PC
    else
        w = mean(clp0, 1)'; % initialize with the mean of NOT drift-corrected trace
        w = w/sum(w.^2)^.5; % unit-normalize
    end

    % initial projections of waveform PCs onto 1D vector
    x = gather(clp * w);
    s1 = var(x(x>mean(x))); % initialize estimates of variance for the first
    s2 = var(x(x<mean(x))); % and second gaussian in the mixture of 1D gaussians

    mu1 = mean(x(x>mean(x))); % initialize the means as well
    mu2 = mean(x(x<mean(x)));
    p  = mean(x>mean(x)); % and the probability that a spike is assigned to the first Gaussian

    logp = zeros(numel(isp), 2); % initialize matrix of log probabilities that each spike is assigned to the first or second cluster

    % do 50 pursuit iteration
    is_okay = true;
    logP = nan(50, 1);
    for k = 1:50
        % for each spike, estimate its probability to come from either Gaussian cluster
        logp(:,1) = -1/2*log(s1) - (x-mu1).^2/(2*s1) + log(p);
        logp(:,2) = -1/2*log(s2) - (x-mu2).^2/(2*s2) + log(1-p);

        lMax = max(logp,[],2);
        logp = logp - lMax; % subtract the max for floating point accuracy
        rs = exp(logp); % exponentiate the probabilities

        pval = log(sum(rs,2)) + lMax; % get the normalizer and add back the max
        logP(k) = mean(pval); % this is the cost function: we can monitor its increase

        rs = rs./sum(rs,2); % normalize so that probabilities sum to 1

        p = mean(rs(:,1)); % mean probability to be assigned to Gaussian 1
        mu1 = (rs(:,1)' * x )/sum(rs(:,1)); % new estimate of mean of cluster 1 (weighted by "responsibilities")
        mu2 = (rs(:,2)' * x )/sum(rs(:,2)); % new estimate of mean of cluster 2 (weighted by "responsibilities")

        s1 = (rs(:,1)' * (x-mu1).^2 )/sum(rs(:,1)); % new estimates of variances
        s2 = (rs(:,2)' * (x-mu2).^2 )/sum(rs(:,2));
        
        if s1 == 0 || s2 == 0
            % this means only 1 spike is left in one of the clusters typically,
            % so we break here and avoid doing the split
            if s1 == 0
                s1 = 1e-6;
                is_okay = false;
            end
            if s2 == 0
                s2 = 1e-6;
                is_okay = false;
            end
            break;
        end

        if (k>10 && rem(k,2)==1)
            % starting at iteration 10, we start re-estimating the pursuit direction
            % that is, given the Gaussian cluster assignments, and the mean and variances,
            % we re-estimate w
            StS  = clp' * (clp .* (rs(:,1)/s1 + rs(:,2)/s2))/nSpikes; % these equations follow from the model
            StMu = clp' * (rs(:,1)*mu1/s1 + rs(:,2)*mu2/s2)/nSpikes;

            if rank(StS) < size(StS, 2)
                is_okay = false;
                break;
            end
            w = StMu'/StS; % this is the new estimate of the best pursuit direection
            w = normc(w'); % which we unit normalize
            x = gather(clp * w);  % the new projections of the data onto this direction
        end
    end
    
    % update the probabilities with the final settings so they can be re-calculated for reextraction using the {w, mu, s, p} values stored in splitProj
    logp(:,1) = -1/2*log(s1) - (x-mu1).^2/(2*s1) + log(p);
    logp(:,2) = -1/2*log(s2) - (x-mu2).^2/(2*s2) + log(1-p);
    lMax = max(logp,[],2);
    logp = logp - lMax; % subtract the max for floating point accuracy
    rs = exp(logp); % exponentiate the probabilities
    rs = rs./sum(rs,2); % normalize so that probabilities sum to 1

    ilow = rs(:,1)>rs(:,2); % these spikes are assigned to cluster 1
    plow = mean(rs(ilow,1)); % the mean probability of spikes assigned to cluster 1
    phigh = mean(rs(~ilow,2)); % same for cluster 2
    nremove = min(mean(ilow), mean(~ilow)); % the smallest cluster has this proportion of all spikes


    % did this split fix the autocorrelograms?
    [K, Qi, Q00, Q01, rir] = ccg(ss(ilow), ss(~ilow), 500, dt); %#ok<ASGLU> % compute the cross-correlogram between spikes in the putative new clusters
    Q12 = min(Qi/max(Q00, Q01)); % refractoriness metric 1
    R = min(rir); % refractoriness metric 2

    % if the CCG has a dip, don't do the split.
    % These thresholds are consistent with the ones from merges.
    if Q12<.25 && R<.05 % if both metrics are below threshold.
        nccg = nccg+1; % keep track of how many splits were voided by the CCG criterion
        continue;
    end

    % clp0 is nspikes x 96
    % now decide if the split would result in waveforms that are too similar
    c1  = wPCA * reshape(mean(clp0(ilow,:),1), 3, []); %  the reconstructed mean waveforms for putatiev cluster 1
    c2  = wPCA * reshape(mean(clp0(~ilow,:),1), 3, []); %  the reconstructed mean waveforms for putative cluster 2
    cc = corrcoef(c1, c2); % correlation of mean waveforms
    n1 =sqrt(sum(c1(:).^2)); % the amplitude estimate 1
    n2 =sqrt(sum(c2(:).^2)); % the amplitude estimate 2

    r0 = 2*abs(n1 - n2)/(n1 + n2); % similarity of amplitudes

    % if the templates are correlated, and their amplitudes are similar, stop the split!!!
    if cc(1,2)>.9 && r0<.2
        continue;
    end

    auc = min(plow, phigh);
    if numel(splitauc) < ik || isempty(splitauc{ik})
        splitauc{ik} = auc;
    else
        splitauc{ik} = cat(1, splitauc{ik}, auc);
    end

    % finaly criteria to continue with the split: if the split piece is more than 5% of all spikes,
    % if the split piece is more than 300 spikes, and if the confidences for assigning spikes to
    % both clusters exceeds a preset criterion ccsplit
    if is_okay && nremove > .05 && min(plow,phigh)>ccsplit && min(sum(ilow), sum(~ilow))>300
        split_candidate(ik) = true;
        if markSplitsOnly
            continue;
        end
        
        % store the splitting axes too for reextraction
        proj = struct('w', w, 'mu1', mu1, 'mu2', mu2, 's1', s1, 's2', s2, 'p', p);
        if numel(splitProj) < ik || isempty(splitProj{ik})
            splitProj{ik} = proj;
        else
            splitProj{ik} = cat(1, splitProj{ik}, proj);
        end

        % actually do the split on the template, one template stays, one goes
        Nfilt = Nfilt + 1;

        ch = rez.iNeighPC(:, ik); % which channels do we overwrite, according to which channels the projections from cProjPC were originally defined on (which is where c1, c2 come from)

        % the templates for the splits have been estimated from PC coefficients
        rez.dWU(:,ch,Nfilt) = c2; % nt0 x NchanNear
        rez.dWU(:,ch,ik)    = c1; % nt0 x NchanNear

         % the temporal components are therefore just the PC waveforms
        rez.W(:,Nfilt,:) = permute(wPCA, [1 3 2]);
        iW(Nfilt) = iW(ik);  % copy the best channel from the original template
        isplit(Nfilt) = isplit(ik); % copy the provenance index to keep track of splits

        split_orig_template(Nfilt) = split_orig_template(ik);
        split_candidate(Nfilt) = false;

        % we change the template assignments in column template_col
        rez.st3(isp(ilow), [template_col, cluster_col])    = Nfilt; % overwrite spike indices with the new index, both in template_col and cluster_col
        rez.simScore(:, Nfilt)   = rez.simScore(:, ik); % copy similarity scores from the original
        rez.simScore(Nfilt, :)   = rez.simScore(ik, :); % copy similarity scores from the original
        rez.simScore(ik, Nfilt) = 1; % set the similarity with original to 1
        rez.simScore(Nfilt, ik) = 1; % set the similarity with original to 1

        rez.iNeigh(:, Nfilt)     = rez.iNeigh(:, ik); % copy neighbor template list from the original
        rez.iNeighPC(:, Nfilt)     = rez.iNeighPC(:, ik); % copy neighbor channel list from the original

        rez.iNeigh(:, Nfilt)    = rez.iNeigh(:, ik);
        rez.iNeighPC(:, Nfilt)  = rez.iNeighPC(:, ik);

        % log the split
        splitsrc(Nfilt) = ik;
        if numel(splitdst) < ik || isempty(splitdst{ik})
            splitdst{ik} = Nfilt;
        else
            splitdst{ik} = cat(1, splitdst{ik}, Nfilt);
        end

        % try this cluster again
        ik = ik-1; % the cluster piece that stays at this index needs to be tested for splits again before proceeding
        % the piece that became a new cluster will be tested again when we get to the end of the list
        nsplits = nsplits + 1; % keep track of how many splits we did
    end
end
prog.finish();

if markSplitsOnly
    fprintf('Found %d split candidates, checked %d/%d clusters, nccg %d \n', nnz(split_candidate), ik, Nfilt, nccg);
else
    fprintf('Finished with %d splits, checked %d/%d clusters, nccg %d \n', nsplits, ik, Nfilt, nccg)
end

rez.split_candidate = split_candidate;

if markSplitsOnly
    splitsrc = nan(Nfilt, 1);
    splitdst = cell(Nfilt, 1);
else
    % zeros get filled in when the array is expanded
    splitsrc(splitsrc == 0) = NaN;
end

Nfilt = size(rez.W,2); % new number of templates
Nrank = 3;
Nchan = ops.Nchan;
Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest ...
    Nrank 0 0 Nchan NchanNear ops.nt0min 0]); % make a new Params to pass on parameters to CUDA

% we need to re-estimate the spatial profiles
[Ka, Kb] = getKernels(ops, 10, 1); % we get the time upsampling kernels again

if reproducible
    [rez.W, rez.U, rez.mu] = mexSVDsmall2r(Params, double(gpuArray(rez.dWU)), double(gpuArray(rez.W)), iC-1, iW-1, double(Ka), double(Kb)); % we run SVD
    
    % round to 6 decimal places
    rez.W = gather(round(rez.W * 2^roundpow2) / 2^roundpow2);
    rez.U = gather(round(rez.U * 2^roundpow2) / 2^roundpow2);
    rez.mu = gather(round(rez.mu * 2^roundpow2) / 2^roundpow2);
else
    [rez.W, rez.U, rez.mu] = mexSVDsmall2(Params, double(gpuArray(rez.dWU)), double(gpuArray(rez.W)), iC-1, iW-1, double(Ka), double(Kb)); % we run SVD
end
[WtW, iList] = getMeWtW(single(rez.W), single(rez.U), Nnearest); % we re-compute similarity scores between templates
rez.iList = iList; % over-write the list of nearest templates

isplit = rez.simScore==1; % overwrite the similarity scores of clusters with same parent
rez.simScore = gather(max(WtW, [], 3));
rez.simScore(isplit) = 1; % 1 means they come from the same parent

rez.iNeigh   = gather(iList(:, 1:Nfilt)); % get the new neighbor templates
rez.iNeighPC    = gather(iC(:, iW(1:Nfilt))); % get the new neighbor channels

prepad = ops.nt0 - 2*ops.nt0min - 1;
rez.Wphy = cat(1, zeros(prepad, Nfilt, Nrank), rez.W); % for Phy, we need to pad the spikes with zeros so the spikes are aligned to the center of the window

rez.isplit = isplit; % keep track of origins for each cluster

% ensure all merge and split arrays end up full size
rez.split_orig_template = split_orig_template;
rez.splitsrc = splitsrc;
rez.splitdst = cat(1, splitdst, cell(Nfilt - numel(splitdst), 1));
rez.splitauc = cat(1, splitauc, cell(Nfilt - numel(splitauc), 1));
rez.splitProjections = cat(1, splitProj, cell(Nfilt - numel(splitProj), 1));
if isfield(rez, 'mergecount')
    rez.mergecount = cat(1, rez.mergecount, zeros(Nfilt - numel(rez.mergecount), 1));
end
if isfield(rez, 'mergedst')
    rez.mergedst = cat(1, rez.mergedst, nan(Nfilt - numel(rez.mergedst), 1));
end

% figure(1)
% subplot(1,4,1)
% plot(logP(1:k))
%
% subplot(1,4,2)
% [~, isort] = sort(x);
% epval = exp(pval);
% epval = epval/sum(epval);
% plot(x(isort), epval(isort))
%
% subplot(1,4,3)
% ts = linspace(min(x), max(x), 200);
% xbin = hist(x, ts);
% xbin = xbin/sum(xbin);
%
% plot(ts, xbin)
%
% figure(2)
% plotmatrix(v(:,1:4), '.')
%
% drawnow
%
% % compute scores for splits
% ilow = rs(:,1)>rs(:,2);
% ps = mean(rs(:,1));
% [mean(rs(ilow,1)) mean(rs(~ilow,2)) max(ps, 1-ps) min(mean(ilow), mean(~ilow))]
