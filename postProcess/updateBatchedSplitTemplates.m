function rez = updateBatchedSplitTemplates(rez)
% based on how splitAllClusters split templates, update rez.dWU, .W_a, W_b, U_a, 

ops = rez.ops;
[nt0, Nfilt, Nrank] = size(rez.W);
Nchan = ops.Nchan;
nKeep = size(rez.W_a, 2); % how many PCs to keep
Nfilt_new = Nfilt - size(rez.WA, 2);

if Nfilt_new == 0
    return;
end

wPCA = gather(ops.wPCA);

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

template_mask = ~isnan(rez.splitdst) | ~isnan(rez.splitsrc);

sz = size(rez.dWUA);
dWUA = rez.dWUA;
rez.dWUA = zeros([sz(1:2), Nfilt, nBatches], 'single');
WA_old = cat(2, gather(rez.WA), repmat(permute(wPCA, [1 3 2]), [1, Nfilt_new, 1, nBatches])); % not on GPU

rez.WA = zeros([nt0, Nfilt, Nrank, nBatches], 'single');
rez.UA = zeros([Nchan, Nfilt, Nrank, nBatches], 'single');
rez.muA = zeros(Nfilt, nBatches, 'single');

% perform efficient cluster-wise, batch-wise averaging of spike-feature projections in cProjPC
batch = isortbatches(rez.st3(:, 5)); % this spike belongs to the batch at this index of isortbatches
template = rez.st3(:, 6); % this spike belongs to this template

prog = ProgressBar(nBatches, 'Updating batchwise, template-wise feature proj averages...\n');

% cProjPC is Nfilt x Nrank x NchanNear. We want to average those rows belonging to template iT and batch iB
% and store them as batch_cluster_avgs(:, :, iT, iB). for this, subs has size [size(cProjPC, 1), 4]
[subsT, subsR, subsC] = ndgrid(template, 1:Nrank, 1:NchanNear);
subsB = repmat(batch, [1 Nrank NchanNear]);
subs = uint32([subsR(:), subsC(:), subsT(:) subsB(:)]); % size [size(cProjPC, 1), 4], specifying rank / ch (indexing into cProjPC dims 2 and 3, and then template and batch
vals = rez.cProjPC;
batch_cluster_sums = accumarray(subs, vals(:), [Nrank, NchanNear, Nfilt, nBatches]);
    
subs = [template, batch];
batch_cluster_counts = shiftdim(accumarray(subs, 1, [Nfilt, nBatches]), -2);

batch_cluster_avgs = batch_cluster_sums ./ batch_cluster_counts;
batch_cluster_avgs(isnan(batch_cluster_avgs)) = 0;

pc_batch_cluster_avgs = reshape(wPCA * reshape(batch_cluster_avgs, [Nrank, NchanNear*Nfilt * nBatches]), ...
    [nt0, NchanNear, Nfilt, nBatches]);

for iB = 1:nBatches
    dWU_batch = gpuArray(cat(3, dWUA(:,:,:,iB), zeros([sz(1:2), Nfilt-size(rez.dWUA, 3)], 'single'))); 
    for ik = 1:Nfilt  
        % ~(this template sourced a split to another template) && ~(this template was split off from another template)
        if ~template_mask(ik)
            continue;
        end
        % apply the split to the batch-wise clusters
        ch = rez.iNeighPC(:, ik);
        dWU_batch(:,ch, ik) = pc_batch_cluster_avgs(:, :, ik, iB); % 61x 32
    end
    

    [WA_b, UA_b, muA_b] = mexSVDsmall2(Params, double(dWU_batch), double(WA_old(:,:,:,iB)), iC-1, iW-1, Ka, Kb);
    rez.WA(:,:,:,iB) = gather(single(WA_b));
    rez.UA(:,:,:,iB) = gather(single(UA_b));
    rez.muA(:,iB) = gather(single(muA_b));
    rez.dWUA(:, :, :, iB) = gather(dWU_batch);
    
    prog.update(iB);
end
prog.finish();

fprintf('Updating decomposition of batchwise W and U\n');
rez.W_a = zeros(nt0 * Nrank, nKeep, Nfilt, 'single');
rez.W_b = zeros(nBatches, nKeep, Nfilt, 'single');
rez.U_a = zeros(Nchan* Nrank, nKeep, Nfilt, 'single');
rez.U_b = zeros(nBatches, nKeep, Nfilt, 'single');
for j = 1:Nfilt
    WA = reshape(rez.WA(:, j, :, :), [], nBatches);
    WA = gpuArray(WA);
    [A, B, C] = svdecon(WA);
    rez.W_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.W_b(:,:,j) = gather(C(:, 1:nKeep));
    
    UA = reshape(rez.UA(:, j, :, :), [], nBatches);
    UA = gpuArray(UA);
    [A, B, C] = svdecon(UA);
    rez.U_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.U_b(:,:,j) = gather(C(:, 1:nKeep));
end

rez.batch_cluster_counts = batch_cluster_counts;




