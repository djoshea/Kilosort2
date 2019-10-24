function rez = updateBatchedSplitTemplates(rez)
% based on how splitAllClusters split templates, update rez.dWU, .W_a, W_b, U_a, 

ops = rez.ops;
roundpow2 = 17;
reproducible = getOr(ops, 'reproducible', false);

[nt0, Nfilt, Nrank] = size(rez.W);
Nchan = ops.Nchan;
nKeep = size(rez.W_a, 2); % how many PCs to keep
% Nfilt_new = Nfilt - size(rez.WA, 2);
% Nfilt_orig = size(rez.WA, 2);

% if Nfilt_new == 0
%     return;
% end

wPCA = gather(ops.wPCA);
wPCAd = double(wPCA);

NchanNear   = min(ops.Nchan, 32);
Nnearest    = min(ops.Nchan, 32);
sigmaMask   = ops.sigmaMask;

% for batchwise updating of templates
nBatches = rez.ops.Nbatch;
isortbatches = rez.iorig(:);

[iC, ~, ~] = getClosestChannels(rez, sigmaMask, NchanNear);
[~, iW] = max(abs(rez.dWU(ops.nt0min, :, :)), [], 2);
iW = squeeze(int32(iW));

Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest Nrank 0 0 Nchan NchanNear ops.nt0min 0]);
[Ka, Kb] = getKernels(ops, 10, 1);

% template_mask = ~isnan(rez.splitdst) | ~isnan(rez.splitsrc);
W_start_nhalf = rez.W;

rez.WA_split = zeros([nt0, Nfilt, Nrank, nBatches], 'single');
rez.UA_split = zeros([Nchan, Nfilt, Nrank, nBatches], 'single');
rez.muA_split = zeros(Nfilt, nBatches, 'single');

% perform efficient cluster-wise, batch-wise averaging of spike-feature projections in cProjPC
batch = isortbatches(rez.st3(:, 5)); % this spike belongs to the batch at this index of isortbatches
template = rez.st3(:, 2); % this spike belongs to this template

prog = ProgressBar(nBatches, 'Updating batchwise, template-wise feature proj averages...\n');

% cProjPC is Nfilt x Nrank x NchanNear. We want to average those rows belonging to template iT and batch iB
% and store them as batch_cluster_avgs(:, :, iT, iB). for this, subs has size [size(cProjPC, 1), 4]
[subsT, subsR, subsC] = ndgrid(template, 1:Nrank, 1:NchanNear);
subsB = repmat(batch, [1 Nrank NchanNear]);
subs = uint32([subsR(:), subsC(:), subsT(:) subsB(:)]); % size [size(cProjPC, 1), 4], specifying rank / ch (indexing into cProjPC dims 2 and 3, and then template and batch
vals = rez.cProjPC;
batch_cluster_sums = accumarray(subs, vals(:), [Nrank, NchanNear, Nfilt, nBatches]);
% batch_cluster_sums = accumarray(subs, vals(:), [Nrank, NchanNear, Nfilt, nBatches], @(x) sum(x, 'double'));
    
subs = [template, batch];
batch_cluster_counts = shiftdim(accumarray(subs, 1, [Nfilt, nBatches]), -2);

batch_cluster_avgs = double(batch_cluster_sums) ./ batch_cluster_counts;
batch_cluster_avgs(isnan(batch_cluster_avgs)) = 0;

pc_batch_cluster_avgs = reshape(wPCAd * reshape(batch_cluster_avgs, [Nrank, NchanNear*Nfilt * nBatches]), ...
    [nt0, NchanNear, Nfilt, nBatches]);

% function to compute single batch dWU
dWU_batch = zeros([nt0, Nchan, Nfilt], 'double', 'gpuArray');
function computeUpdateSingleBatch(ibatch)
    dWU_batch(:) = 0;
    for ik = 1:Nfilt  
        % apply the split to the batch-wise clusters
        ch = rez.iNeighPC(:, ik);
        dWU_batch(:,ch, ik) = pc_batch_cluster_avgs(:, :, ik, ibatch); % 61x 32
    end
    if reproducible
        dWU_batch = round(dWU_batch * 2^roundpow2) / 2^roundpow2;
    end
end

% loop over batches in the same pattern from the middle outward
% as is used in learnAndSolve8b
% and enfoce smoothness in dWU according to the final annealing schedule
nhalf = ceil(nBatches/2);
ischedule = [nhalf:-1:1, nhalf:nBatches];

pmi = exp(-1./ops.momentum(2));

for iS = 1:numel(ischedule)
    korder = ischedule(iS); % korder is the index of the batch at this point in the schedule
    ibatch = isortbatches(korder); % ibatch is the index of the batch in absolute terms
    
    % this will efficiently update dWU_batch in place
    computeUpdateSingleBatch(ibatch), 

    if korder == nhalf
        % use the actual estimate for the middle batch
        dWU_smoothed = rez.dWU;
        W_start = W_start_nhalf;
    else
        % smooth from the previous batch
        cluster_counts_this = double(batch_cluster_counts(:, :, :, ibatch));
        fexp = exp(cluster_counts_this.*log(pmi));
        dWU_smoothed = dWU_smoothed .* fexp + (1-fexp) .* (dWU_batch./max(1,cluster_counts_this));
        
        korder_prev = ischedule(iS-1);
        W_start = rez.WA_split(:, :, :, isortbatches(korder_prev));
    end

    if reproducible
        [WA_b, UA_b, muA_b] = mexSVDsmall2r(Params, double(dWU_smoothed), double(W_start), iC-1, iW-1, Ka, Kb);
    else
        [WA_b, UA_b, muA_b] = mexSVDsmall2(Params, double(dWU_smoothed), double(W_start), iC-1, iW-1, Ka, Kb);
    end
    rez.WA_split (:, :, :, ibatch) = gather(single(WA_b));
    rez.UA_split (:, :, :, ibatch) = gather(single(UA_b));
    rez.muA_split(:, ibatch) = gather(single(muA_b));
%     rez.dWUA(:, :, :, iB) = gather(dWU_batch);
    
    prog.update(iS);
end
prog.finish();

fprintf('Updating decomposition of batchwise W and U\n');
rez.W_a_split = zeros(nt0 * Nrank, nKeep, Nfilt, 'single');
rez.W_b_split = zeros(nBatches, nKeep, Nfilt, 'single');
rez.U_a_split = zeros(Nchan* Nrank, nKeep, Nfilt, 'single');
rez.U_b_split = zeros(nBatches, nKeep, Nfilt, 'single');
for j = 1:Nfilt
    WA = reshape(rez.WA_split(:, j, :, :), [], nBatches);
    WA = gpuArray(WA);
    [A, B, C] = svdecon(WA);
    rez.W_a_split(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.W_b_split(:,:,j) = gather(C(:, 1:nKeep));
    
    UA = reshape(rez.UA_split(:, j, :, :), [], nBatches);
    UA = gpuArray(UA);
    [A, B, C] = svdecon(UA);
    rez.U_a_split(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.U_b_split(:,:,j) = gather(C(:, 1:nKeep));
end

rez.batch_cluster_counts = batch_cluster_counts;


end
