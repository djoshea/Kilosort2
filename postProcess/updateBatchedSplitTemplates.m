function rez = updateBatchedSplitTemplates(rez)
% based on how splitAllClusters split templates, update rez.dWU, .W_a, W_b, U_a, 
ops = rez.ops;
wPCA = gather(ops.wPCA);

NchanNear   = min(ops.Nchan, 32);
Nnearest    = min(ops.Nchan, 32);
sigmaMask   = ops.sigmaMask;

% for batchwise updating of templates
nBatches = rez.ops.Nbatch;
isortbatches = rez.iorig(:);
[~, batch_sort_position] = ismember(1:nBatches, isortbatches); % for data batch x, batch_sort_position(x) gives the value of st3(:, 5) corresponding to those spikes

[nt0, Nfilt, Nrank] = size(rez.W);
Nchan = ops.Nchan;
nKeep = size(rez.W_a, 2); % how many PCs to keep

[iC, ~, ~] = getClosestChannels(rez, sigmaMask, NchanNear);
[~, iW] = max(abs(rez.dWU(ops.nt0min, :, :)), [], 2);
iW = squeeze(int32(iW));

Params     = double([0 Nfilt 0 0 size(rez.W,1) Nnearest Nrank 0 0 Nchan NchanNear ops.nt0min 0]);
[Ka, Kb] = getKernels(ops, 10, 1);

template_mask = ~isnan(rez.splitdst) | ~isnan(rez.splitsrc);

sz = size(rez.dWUA);

Nfilt_new = Nfilt - size(rez.WA, 2);
WA_old = cat(2, gather(rez.WA), repmat(permute(wPCA, [1 3 2]), [1, Nfilt_new, 1, nBatches])); % not on GPU

rez.WA = zeros([nt0, Nfilt, Nrank, nBatches], 'single');
rez.UA = zeros([Nchan, Nfilt, Nrank, nBatches], 'single');
rez.muA = zeros(Nfilt, nBatches, 'single');

for iB = 1:nBatches
    dWU_batch = gpuArray(cat(3, rez.dWUA(:,:,:,iB), zeros([sz(1:2), Nfilt-size(rez.dWUA, 3)], 'single')));

    mask_batch = rez.st3(:, 5) == batch_sort_position(iB);
    template_batch = rez.st3(mask_batch, 6);
    [subs1, subs2, subs3] = ndgrid(template_batch, 1:Nrank, 1:NchanNear);
    subs = [subs1(:), subs2(:), subs3(:)];
    vals = rez.cProjPC(mask_batch,:,:);
    avgs = accumarray(subs, vals(:), [Nfilt, Nrank, NchanNear], @mean, single(0));
    
    for ik = 1:Nfilt  
        % ~(this template sourced a split to another template) && ~(this template was split off from another template)
        if ~template_mask(ik)
            continue;
        end
        % apply the split to the batch-wise clusters
        avg_this = avgs(ik, :, :);  
        c_batch  = wPCA * reshape(avg_this, 3, []);
        dWU_batch(:,iC(:, iW(ik)), ik)    = c_batch; % 61x 32
    end

    [WA_b, UA_b, muA_b] = mexSVDsmall2(Params, dWU_batch, gpuArray(WA_old(:,:,:,iB)), iC-1, iW-1, Ka, Kb);
    rez.WA(:,:,:,iB) = gather(WA_b);
    rez.UA(:,:,:,iB) = gather(UA_b);
    rez.muA(:,iB) = gather(muA_b);
end

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




