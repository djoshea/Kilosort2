function out = reextractSpikesWithFixedTemplates(ks, varargin)
% the goal here is to re-do the spike extraction phase of Kilosort2 for specific windows of time, where we keep all the 
% templates fixed as they were when KS2 was run on the file, but overwriting the data in these specific windows either 
% with different data or by zeroing certain samples with a mask.  
% 
% replace is a nChSorted x nTime sparse matrix of data values that will overwrite the data in the raw imec file
% extractTimeMask is a nTime length sparse logical indicating during which times spikes extracted should be returned
% 
% this will load back in our data to RAM (if ops.useRAM) or re-write it to the ops.fproc file

p = inputParser();
p.addParameter('DATA', [], @(x) true);
p.addParameter('batch_inds', [], @(x) true); % manually specify list / mask of batches to resort, if empty, all batches are used

% these are used to replace specific time windows in the raw data with 
p.addParameter('data_replace_windows', zeros(0, 2), @(x) ismatrix(x) && size(x, 2) == 2);
p.addParameter('data_replace', {}, @iscell);

% if true, only spikes that occur within (data_replace_windows(:, 1) - pad_data_replace_windows(1)) : (data_replace_windows(:, 2) + pad_data_replace_windows(2)) will be returned 
p.addParameter('spike_extract_windows', zeros(0, 2), @(x) ismatrix(x) && size(x, 2) == 2);
p.parse(varargin{:});
spike_extract_windows = p.Results.spike_extract_windows;

ks.load();

msg = 'KilosortDataset needs to have preSplit fields loaded to reextract spikes';
assert(ks.hasPreSplitLoaded, msg);
assert(~isempty(ks.W_preSplit), msg);
assert(~isempty(ks.U_preSplit), msg);
assert(~isempty(ks.iW_preSplit), msg);
assert(~isempty(ks.W_batch_preSplit), msg);
assert(~isempty(ks.U_batch_preSplit), msg);
assert(~isempty(ks.mu_batch_preSplit), msg);

ops = ks.ops;
ops.chanMap = fullfile(ops.root,'chanMap.mat');

% goal here is to load or re-load data such that batch_inds will match DATA along dim 3
if isempty(p.Results.DATA) 
     % if isempty(spike_extract_windows) all batches will be loaded
     % otherwise only those batches with overlap will be loaded
    rez = preprocessDataSub(ops, 'batch_inds', p.Results.batch_inds, ...
        'data_replace_windows', p.Results.data_replace_windows, 'data_replace', p.Results.data_replace, ...         
        'only_batches_overlapping_windows', spike_extract_windows);
    %assignin('base', 'rez_re', rez);
    ops = rez.ops;
    DATA = rez.DATA;
    if isfield(rez, 'DATA_batch_inds')
        batch_inds = rez.DATA_batch_inds;
    else
        batch_inds = 1:ops.Nbatch;
    end
   
else
    if ~isempty(p.Results.data_replace)
        warning('Assuming that DATA passed into reextractSpikesWithFixedTemplates has been pre-modified with %d data_replace segments', numel(p.Results.data_replace));
    end
    % preloaded DATA with no data_replace, okay to use as is and treat batch_inds as the associated batches if specified
    DATA = p.Results.DATA;
    if isempty(p.Results.batch_inds)
        batch_inds = 1:ops.Nbatch;
    else
        batch_inds = p.Results.batch_inds;
    end
    
    ops.useRAM = true;
    
    % needed for getClosestChannels below
    coords = ks.channel_positions_sorted;
    rez.xc = coords(:, 1);
    rez.yc = coords(:, 2);
end

assert(numel(batch_inds) == size(DATA, 3));
nBatchesResort = numel(batch_inds);

NrankPC = 6;  
Nrank = 3;
rng('default'); rng(1);

NchanNear   = min(ops.Nchan, 32);
Nnearest    = min(ops.Nchan, 32);

sigmaMask  = ops.sigmaMask;
nt0 = ops.nt0;
nt0min  = ops.nt0min; 

nBatches  = ks.nBatches;
NT  	= ops.NT;
batchstart = 0:NT:NT*nBatches;
Nchan 	= ops.Nchan;

[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear); %#ok<ASGLU>
t0 = ceil(ops.trange(1) * ops.fs);

nInnerIter  = 60;
pmi_end = exp(-1./ops.momentum(2));

% importantly, we use the batchwise templates computed during learnAndSolve8b, 
% i.e. before any splitting is performed. 
Nfilt = ks.nTemplatesPreSplit; 

Nsum = 7; % how many channels to extend out the waveform in mexgetspikes
Params     = double([NT Nfilt ops.Th(end) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi_end Nchan NchanNear ops.nt0min 2 Nsum NrankPC ops.Th(1)]);
% Params(3) set to ops.Th(end) --> different threshold on extraction pass
% Params(9) set to pmi(end) == exp(-1/ops.momentum(2))
% Params(13) set to 2  --> extract ALL features on the extraction pass

if ~ops.useRAM
    fid = fopen(ops.fproc, 'r');
end

% normally KS2 would extract dWU from the data, then decompose into W, U, and mu
% here we reconstruct approximate dWU from W, U, and mu
W = ks.W_preSplit;
U = ks.U_preSplit;
iW = ks.iW_preSplit;

[WtW, iList] = getMeWtW(single(W), single(U), Nnearest); %#ok<ASGLU>
wPCA = ops.wPCA;

% first iteration of final extraction pass
prog = ProgressBar(nBatchesResort, 'Re-extracting modified batches');

ntot = 0;
st3 = zeros(1e7, 5);
fW  = zeros(Nnearest, 1e7, 'single');
fWpc = zeros(NchanNear, Nrank, 1e7, 'single');

for iibatch = 1:nBatchesResort
    ibatch = batch_inds(iibatch);
    
    ioffset = ops.ntbuff;
    if ibatch==1
        ioffset = 0;
    end
    toff = nt0min + t0 - ioffset + (NT-ops.ntbuff)*(ibatch-1);
    
    if ~ops.useRAM
        offset = 2 * ops.Nchan*batchstart(ibatch);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
        dat = DATA(:, :, iibatch); % use iibatch since DATA dim 3 already matches batch_inds
    end

    dataRAW = single(gpuArray(dat))/ ops.scaleproc;

    Params(1) = size(dataRAW, 1); % update NT each loop in case we subset dataRAW
    
    W = ks.W_batch_preSplit(:, :, :, ibatch);
    U = ks.U_batch_preSplit(:, :, :, ibatch);
    mu = ks.mu_batch_preSplit(:, ibatch);
    
    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); %#ok<ASGLU>
   
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp] = ...
        mexMPnu8r(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA); %#ok<ASGLU>
   
    st = toff + double(st0);
    if ~isempty(spike_extract_windows)
        % filter st to only include those that occur in a data replace windows
        mask_st = any(st(:) >= spike_extract_windows(:, 1)' & st(:) <= spike_extract_windows(:, 2)', 2); % nSpikes x nWindows --> nSpikes
        st = st(mask_st);
        id0 = id0(mask_st);
        x0 = x0(mask_st);
        vexp = vexp(mask_st);
        featW = featW(:, mask_st);
        featPC = featPC(:, :, mask_st);
    end
    
    % double the buffer size if needed
    if ntot+numel(x0)>size(st3,1)
       fW(:, 2*size(st3,1))    = 0;
       fWpc(:,:,2*size(st3,1)) = 0;
       st3(2*size(st3,1), 1)   = 0;
    end
    
    % append new spikes to st3, fW, fWpc
    irange = ntot + (1:numel(x0));
    st3(irange,1) = double(st);
    st3(irange,2) = double(id0+1);
    st3(irange,3) = double(x0);
    st3(irange,4) = double(vexp);
    st3(irange,5) = find(ibatch == ks.batch_sort_order);

    fW(:, irange) = gather(featW);
    fWpc(:, :, irange) = gather(featPC);

    ntot = ntot + numel(x0);
    prog.update(iibatch);
end
prog.finish();

if ~ops.useRAM
    fclose(fid);
end

st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

[st3, sortIdx] = sortrows(st3);
fW = fW(:, sortIdx);
fWpc = fWpc(:, :, sortIdx);

cProj = fW';
cProjPC = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

out.st3 = st3;
out.cProj = cProj;
out.cProjPC = cProjPC;

% copy orig templates to cols 6 and 7 to hold updated templates and clusters modifed by merge and split
out.st3(:, 6) = out.st3(:, 2);
out.st3(:, 7) = out.st3(:, 2);
out.st3_template_col = 6;
out.st3_cluster_col = 7;

% next, apply splits to templates using existing projection weights
if getOr(ops, 'splitThenMerge', false)
    out = reapplySplits(out, ks);
    out = reapplyMerges(out, ks);
else
    out = reapplyMerges(out, ks);
    out = reapplySplits(out, ks);
end
out = reapplyCutoffs(out, ks);

end

function out = reapplySplits(out, ks)
    template_col = out.st3_template_col;
    cluster_col = out.st3_cluster_col;
    Nfilt = ks.nTemplates;
    prog = Neuropixel.Utils.ProgressBar(Nfilt, 'Reapplying cluster splitting');
    for iT = 1:Nfilt
        split_to_list = ks.cluster_split_dst{iT} + 1;  % these will be 0 indexed as clusters, convert to 1 indexed for out
        split_proj_list = ks.cluster_split_projections{iT};
        for iS = 1:numel(split_to_list)
            split_to = split_to_list(iS);
            proj = split_proj_list(iS);
            isp = find(out.st3(:, cluster_col)==iT);

            clp0 = out.cProjPC(isp, :, :); % get the PC projections for these spikes
            clp0 = clp0(:,:);
            clp = clp0 - mean(clp0,1); % mean center them
            clp = clp - my_conv2(clp, 250, 1); % subtract a running average, because the projections are NOT drift corrected


            % initial projections of waveform PCs onto 1D vector
            x = gather(clp * proj.w);
            logp = zeros(numel(x), 2); % initialize matrix of log probabilities that each spike is assigned to the first or second cluster

            % for each spike, estimate its probability to come from either Gaussian cluster
            logp(:,1) = -1/2*log(proj.s1) - (x-proj.mu1).^2/(2*proj.s1) + log(proj.p);
            logp(:,2) = -1/2*log(proj.s2) - (x-proj.mu2).^2/(2*proj.s2) + log(1-proj.p);

            lMax = max(logp,[],2);
            logp = logp - lMax; % subtract the max for floating point accuracy
            rs = exp(logp); % exponentiate the probabilities
            rs = rs./sum(rs,2); % normalize so that probabilities sum to 1
            ilow = rs(:,1)>rs(:,2); % these spikes will be re-assigned to split_to
            out.st3(isp(ilow), [template_col, cluster_col]) = split_to; % overwrite cluster and template with the new assignment 
        end
        prog.update(iT);
    end
    prog.finish();
end

function out = reapplyMerges(out, ks)
    cluster_col = out.st3_cluster_col;
    Nfilt = ks.nTemplates;
    prog = Neuropixel.Utils.ProgressBar(Nfilt, 'Reapplying cluster merging');
    for iT = 1:Nfilt
        merge_dst = getFinalMergeDest(iT); % these will be 0 indexed as clusters, convert to 1 indexed for out
        if isnan(merge_dst)
            continue;
        end
        
        mask = out.st3(:, cluster_col) == iT;
        out.st3(mask, cluster_col) = merge_dst;
        prog.update(iT);
    end
    prog.finish();
    
    function dest = getFinalMergeDest(src)
        dest = ks.cluster_merge_dst(src) + 1; % 0-indexing --> 1-indexing
        next_dest = dest;
        while ~isnan(next_dest)
            % was dest merged into another cluster?
            dest = next_dest;
            next_dest = ks.cluster_merge_dst(dest) + 1; % 0-indexing --> 1-indexing
        end
    end
end

function out = reapplyCutoffs(out, ks)
    Nfilt = ks.nTemplates;
    vexp = out.st3(:, 4);
    cluster = out.st3(:, out.st3_cluster_col);
    valid_mask = true(numel(vexp), 1);
    
    prog = Neuropixel.Utils.ProgressBar(Nfilt, 'Reapplying threshold cutoffs');
    for iFilt = 1:Nfilt
        th = ks.cutoff_thresholds(iFilt);
        cluster_mask = cluster == iFilt;
        valid_mask(cluster_mask) = vexp(cluster_mask) > th;
        prog.update(iFilt);
    end

    % copy to invalid fields
    if isfield(out, 'st3_cutoff_invalid')
        out.st3_cutoff_invalid = cat(1, out.st3_cutoff_invalid, out.st3(~valid_mask, :));
        out.cProj_cutoff_invalid = cat(1, out.cProj_cutoff_invalid, out.cProj(~valid_mask, :));
        out.cProjPC_cutoff_invalid = cat(1, out.cProjPC_cutoff_invalid, out.cProjPC(~valid_mask, :, :));
    else
        out.st3_cutoff_invalid = out.st3(~valid_mask, :);
        out.cProj_cutoff_invalid = out.cProj(~valid_mask, :);
        out.cProjPC_cutoff_invalid = out.cProjPC(~valid_mask, :, :);
    end
    
    % and delete them from primary
    out.st3(~valid_mask, :) = [];
    out.cProj(~valid_mask, :) = []; % remove their template projections too
    out.cProjPC(~valid_mask, :,:) = [];  % and their PC projections
end
