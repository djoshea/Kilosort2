function rez = learnAndSolve8b(rez)
% This is the main optimization. Takes the longest time and uses the GPU heavily.

% isequaltol = @(a, b) isequal(size(a), size(b)) && max(abs(a - b), [], 'all') < 1e-5;

ops = rez.ops;
ops.fig = getOr(ops, 'fig', 1); % whether to show plots every N batches

reproducible = getOr(ops, 'reproducible', false);
roundpow2 = 17;

NrankPC = 6; % this one is the rank of the PCs, used to detect spikes with threshold crossings
Nrank = 3; % this one is the rank of the templates
rng('default'); rng(1);
if reproducible
    gpurng('default'); gpurng(1);
end

% we need PC waveforms, as well as template waveforms
[wTEMP, wPCA]    = extractTemplatesfromSnippets(rez, NrankPC, false); % distrusted, RAM

% move these to the GPU
wPCA = gpuArray(wPCA(:, 1:Nrank));
wTEMP = gpuArray(wTEMP);
wPCAd = double(wPCA); % convert to double for extra precision
ops.wPCA = gather(wPCA);
ops.wTEMP = gather(wTEMP);
nt0 = ops.nt0;
nt0min  = rez.ops.nt0min;
rez.ops = ops;
nBatches  = rez.temp.Nbatch;
NT  	= ops.NT;
Nfilt 	= ops.Nfilt;
Nchan 	= ops.Nchan;

% two variables for the same thing? number of nearest channels to each primary channel
NchanNear   = min(ops.Nchan, 32);
Nnearest    = min(ops.Nchan, 32);

% decay of gaussian spatial mask centered on a channel
sigmaMask  = ops.sigmaMask;
% spike threshold for finding missed spikes in residuals
% ops.spkTh = -6; % why am I overwriting this here?

batchstart = 0:NT:NT*nBatches;

% find the closest NchanNear channels, and the masks for those channels
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);

% sorting order for the batches
isortbatches = rez.iorig(:);
nhalf = ceil(nBatches/2); % halfway point

% modified schedule from original KS2 to do a full pass through the data
modified_schedule = getOr(ops, 'use_modified_schedule', true);
if modified_schedule
    ischedule = [1:nBatches nBatches:-1:nhalf];
    i1 = (nhalf-1):-1:1;
    i2 = nhalf:nBatches;

    iter_revert_middle = numel(ischedule) + numel(i1) + 1; % halfway through final pass, revert back to the memorized W for middle batch
    iter_finalize = numel(ischedule); % finalize and memorizeW
    iter_final_pm = nBatches; % finalize annealing parameter pm using final momentum
else
    % this batch order schedule goes through half of the data forward and backward during the model fitting
    % and then goes through the data symmetrically-out from the center during the final pass
    ischedule = [nhalf:nBatches nBatches:-1:nhalf];
    i1 = (nhalf-1):-1:1;
    i2 = nhalf:nBatches;

    iter_revert_middle = nBatches + nhalf + 1; % should be >iter-nBatches where korder ==  nhalf
    iter_finalize = nBatches + 1; % should match niter-nBatches
    iter_final_pm = nBatches + 1; % should match niter-nBatches
end


irounds = cat(2, ischedule, i1, i2); % this is the way we traverse the sorted-batch-list, second half, back to center

niter   = numel(irounds);
% if irounds(niter - nBatches)~=nhalf
%     error('mismatch between number of batches'); % this check is in here in case I do somehting weird when I try different schedules
% end

% these two flags are used to keep track of what stage of model fitting we're at
% flag_final    =  0;
flag_resort   = 1;

% this is the absolute temporal offset in seconds corresponding to the start of the
% spike sorted time segment
t0 = ceil(rez.ops.trange(1) * ops.fs);

nInnerIter  = 60; % this is for SVD for the power iteration

% schedule of learning rates for the model fitting part
% starts small and goes high, it corresponds approximately to the number of spikes
% from the past that were averaged to give rise to the current template
pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), iter_final_pm));

Nsum = min(Nchan,7); % how many channels to extend out the waveform in mexgetspikes
% lots of parameters passed into the CUDA scripts
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ops.nt0min 1 Nsum NrankPC ops.Th(1)]);

% W0 has to be ordered like this
W0 = permute(wPCAd, [1 3 2]);

% initialize the list of channels each template lives on
iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

% initialize average number of spikes per batch for each template
nsp = gpuArray.zeros(0,1, 'double');

% this flag starts 0, is set to 1 later
Params(13) = 0;

% kernels for subsample alignment
[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate in each batch

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

if ~ops.useRAM
    fid = fopen(ops.fproc, 'r');
end

ntot = 0;
ndrop = zeros(1,2); % this keeps track of dropped templates for debugging purposes

m0 = ops.minFR * ops.NT/ops.fs; % this is the minimum firing rate that all templates must maintain, or be dropped

prog = ProgressBar(niter, 'Performing initial pass over batches');
for ibatch = 1:niter
    % irounds is the list of indices we traverse through sorted batch list (isortbatches) --> this loop stored as korder
    % k is the actual batch in the unsorted data (using isortbatches)
    % there are nBatches total batches, and niter = 2*nBatches

    % we loop over batches first for optimization (starting at half going right, then starting at 1 back
    % so irounds goes nhalf --> nBatches, nBatches --> nhalf, nhalf-1 --> 1, nhalf-->nbatches
    %     k = irounds(ibatch);
    korder = irounds(ibatch); % korder is the index of the batch at this point in the schedule
    k = isortbatches(korder); % k is the index of the batch in absolute terms

    %if ibatch>niter-nBatches && korder==nhalf
    if ibatch == iter_revert_middle
        % this is required to revert back to the template states in the middle of the batches
        % halfway through in the final pass, we've reached the end of the

        [W, dWU] = revertW(rez);
        prog.pause_for_output();
        fprintf('Reverted back to middle batch templates\n')
    end

    %if ibatch<=niter-nBatches
    if ibatch <= iter_final_pm
        % obtained pm for this batch
        % in initial pass, load up annealing schedule,
        % the value of pmi(end) will be used for the final extraction pass
        Params(9) = pmi(ibatch);
        pm = pmi(ibatch) * gpuArray.ones(Nfilt, 1, 'double');
    else
        pm = pmi(end) * gpuArray.ones(Nfilt, 1, 'double');
    end

    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ~ops.useRAM
        % loading a single batch (same as everywhere)
        offset = 2 * ops.Nchan*batchstart(k);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
        dat = rez.DATA(:, :, k);
    end
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;

    if ibatch < iter_finalize && ~isempty(rez.distrust_batched)
        % in initial pass or first iteration of final pass where templates are finalized, don't use distrusted samples
        distrust_this_batch = rez.distrust_batched(:, k);
        dataRAW = dataRAW(~distrust_this_batch, :);
    end
    Params(1) = size(dataRAW, 1); % update NT each loop in case we subset dataRAW

    if ibatch==1
        % only on the first batch, we first get a new set of spikes from the residuals,
        % which in this case is the unmodified data because we start with no templates
        % dWU is nt0 x nChan x
        if reproducible
            [dWU, cmap, st, id] = mexGetSpikes2r(Params, dataRAW, wTEMP, iC-1);    %#ok<ASGLU> % CUDA function to get spatiotemporal clips from spike detections
            % @djoshea for reproducibility, sort dWU by st, id
            [~, sortIdx] = sortrows([st, id]);
            dWU = double(dWU(:, :, sortIdx));

            dWU = reshape(wPCAd * (wPCAd' * dWU(:,:)), size(dWU)); % project these into the wPCA waveforms
            dWU = round(dWU * 2^roundpow2) / 2^roundpow2;
        else
            [dWU, cmap] = mexGetSpikes2(Params, dataRAW, wTEMP, iC-1);    %#ok<ASGLU> % CUDA function to get spatiotemporal clips from spike detections

            dWU = double(dWU);
            dWU = reshape(wPCAd * (wPCAd' * dWU(:,:)), size(dWU)); % project these into the wPCA waveforms
        end

        W = W0(:,ones(1,size(dWU,3)),:); % initialize the low-rank decomposition with standard waves
        Nfilt = size(W,2); % update the number of filters/templates
        nsp(1:Nfilt) = m0; % initialize the number of spikes for new templates with the minimum allowed value, so it doesn't get thrown back out right away
        Params(2) = Nfilt; % update in the CUDA parameters
    end

    if flag_resort
        % initial pass, resort the templates by channel lowest to highest
        % iW is Nfilt x 1(templates) vector of channel indices where the biggest deflection is (at time nt0min)
        % this is a flag to resort the order of the templates according to best peak channel
        % this is important in order to have cohesive memory requests from the GPU RAM
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));

        % then we sort the templates by iW, we don't need to resort U as it will be recomputed immediately afterwards
        [iW, isort] = sort(iW); % sort by max abs channel
        W = W(:,isort, :); % user ordering to resort all the other template variables
        dWU = dWU(:,:,isort);
        nsp = nsp(isort);
    end

    % decompose dWU by svd of time and space (via covariance matrix of 61 by 61 samples)
    % this uses a "warm start" by remembering the W from the previous iteration
    if reproducible
        [W, U, mu] = mexSVDsmall2r(Params, double(dWU), double(W), iC-1, iW-1, double(Ka), double(Kb));

        % round to 6 decimal places
        W = round(W * 2^roundpow2) / 2^roundpow2;
        U = round(U * 2^roundpow2) / 2^roundpow2;
        mu = round(mu * 2^roundpow2) / 2^roundpow2;
    else
        [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);
    end

    % UtU is the gram matrix of the spatial components of the low-rank SVDs
    % it tells us which pairs of templates are likely to "interfere" with each other
    % such as when we subtract off a template
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); %#ok<ASGLU>

    % main CUDA function in the whole codebase. does the iterative template matching
    % based on the current templates, gets features for these templates if requested (featW, featPC),
    % gets scores for the template fits to each spike (vexp), outputs the average of
    % waveforms assigned to each cluster (dWU0),
    % and probably a few more things I forget about
    if reproducible
        [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp] = ...
            mexMPnu8r(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
            wPCA);
        % @djoshea: output of mexMPnu8 is reproducible if st0, id0, x0, featW, featPC, and vexp are
        % sorted by [~, sortIdx] = sort(st0) along the appropriate nspikes dimension. The sort order doesn't affect
        % subsequent processing though so it isn't strictly necessary, so we save time by doing the sorting once at end

        % round to # decimal places
        dWU0 = round(dWU0 * 2^roundpow2) / 2^roundpow2;
    else
        [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp] = ...
            mexMPnu8(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
            wPCA);
    end

    % Sometimes nsp can get transposed (think this has to do with it being
    % a single element in one iteration, to which elements are added
    % nsp, nsp0, and pm must all be row vectors (Nfilt x 1), so force nsp
    % to be a row vector.
    [nsprow, nspcol] = size(nsp);
    if nsprow<nspcol
        nsp = nsp';
    end

    % updates the templates as a running average weighted by recency
    % since some clusters have different number of spikes, we need to apply the
    % exp(pm) factor several times, and fexp is the resulting update factor
    % for each template
    fexp = exp(double(nsp0).*log(pm(1:Nfilt)));
    fexp = reshape(fexp, 1,1,[]);
    dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, []));

    % nsp just gets updated according to the fixed factor p1
    nsp = nsp * p1 + (1-p1) * double(nsp0);

    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ibatch==iter_finalize
        % if we reached this point, we need to disable secondary template updates
        % like dropping, and adding new templates. We need to memorize the state of the
        % templates at this timepoint, and set the processing mode to "extraction and tracking"

        prog.update(ibatch, 'Performing final extraction pass over batches');
        % halfway done, so we switch to final extraction mode and memorizeW
        flag_resort   = 0;

        % final clean up, triage templates one last time
        [W, U, dWU, mu, nsp, ndrop] = ...
            triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);

        % final number of templates
        Nfilt = size(W,2);
        Params(2) = Nfilt;

        % final covariance matrix between all templates
        [WtW, iList] = getMeWtW(single(W), single(U), Nnearest);

        % iW is the final channel assigned to each template
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));

        % extract ALL features on the last pass
        Params(13) = 2; % this is a flag to output features (PC and template features)

        % different threshold on last pass?
        Params(3) = ops.Th(end); % usually the threshold is much lower on the last pass

        rez = memorizeW(rez, W, dWU, U, mu);
        prog.pause_for_output();
        fprintf('Memorized middle batch templates\n')
    end

    if ibatch < iter_finalize %-50
        % initial pass through the batches, during the main "learning" phase of fitting a model
        if rem(ibatch, 5)==1
            % this drops templates based on spike rates and/or similarities to other templates
            [W, U, dWU, mu, nsp, ndrop] = ...
                triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);
        end
        Nfilt = size(W,2); % update the number of filters
        Params(2) = Nfilt;

        % this adds new templates if they are detected in the residual
        if reproducible
            [dWU0, cmap, st, id] = mexGetSpikes2r(Params, drez, wTEMP, iC-1); %#ok<ASGLU>
            % @djoshea for reproducibility, sort dWU by st, id
            [~, sortIdx] = sortrows([st, id]);
            dWU0 = double(dWU0(:, :, sortIdx));

            % round to 6 decimal places
            dWU0 = round(dWU0 * 2^roundpow2) / 2^roundpow2;

            if size(dWU0, 3) > 0
                % new templates need to be integrated into the same format as all templates
                dWU0 = reshape(wPCA * (wPCA' * dWU0(:,:)), size(dWU0));
                dWU = cat(3, dWU, dWU0);
            end
        else
            [dWU0, cmap] = mexGetSpikes2(Params, drez, wTEMP, iC-1); %#ok<ASGLU>
            if size(dWU0, 3) > 0
                % new templates need to be integrated into the same format as all templates
                dWU0 = double(dWU0);
                dWU0 = reshape(wPCAd * (wPCAd' * dWU0(:,:)), size(dWU0));
                dWU = cat(3, dWU, dWU0);
            end
        end

        if size(dWU0,3)>0
            W(:,Nfilt + (1:size(dWU0,3)),:) = W0(:,ones(1,size(dWU0,3)),:); % initialize temporal components of waveforms

            nsp(Nfilt + (1:size(dWU0,3))) = ops.minFR * NT/ops.fs; % initialize the number of spikes with the minimum allowed
            mu(Nfilt + (1:size(dWU0,3)))  = 10; % initialize the amplitude of this spike with a lowish number

            Nfilt = min(ops.Nfilt, size(W,2)); % if the number of filters exceed the maximum allowed, clip it

            Params(2) = Nfilt;

            W   = W(:, 1:Nfilt, :); % remove any new filters over the maximum allowed
            dWU = dWU(:, :, 1:Nfilt); % remove any new filters over the maximum allowed
            nsp = nsp(1:Nfilt); % remove any new filters over the maximum allowed
            mu  = mu(1:Nfilt); % remove any new filters over the maximum allowed
        end

    end

    if ibatch > iter_finalize
        % in the final extraction pass, not the first iteration though
        % during the final extraction pass, this keeps track of all spikes and features

        % we memorize the spatio-temporal decomposition of the waveforms at this batch
        % this is currently only used in the GUI to provide an accurate reconstruction
        % of the raw data at this time
%         rez.dWU0A(:,:,:,k) = gather(dWU0);
%         rez.dWUA(:,:,:,k) = gather(dWU);
        rez.WA(:,:,:,k) = gather(W);
        rez.UA(:,:,:,k) = gather(U);
        rez.muA(:,k) = gather(mu);

        % we carefully assign the correct absolute times to spikes found in this batch
        ioffset         = ops.ntbuff;
        if k==1
            ioffset         = 0; % the first batch is special (no pre-buffer)
        end
        toff = nt0min + t0 -ioffset + (NT-ops.ntbuff)*(k-1);
        st = toff + double(st0);
        irange = ntot + (1:numel(x0));

        if ntot+numel(x0)>size(st3,1)
        % if we exceed the original allocated memory, double the allocated sizes
           fW(:, 2*size(st3,1))    = 0;  %#ok<AGROW>
           fWpc(:,:,2*size(st3,1)) = 0;  %#ok<AGROW>
           st3(2*size(st3,1), 1)   = 0;  %#ok<AGROW>
        end

        st3(irange,1) = double(st); %#ok<AGROW> % spike times
        st3(irange,2) = double(id0+1); %#ok<AGROW> % spike clusters (1-indexing)
        st3(irange,3) = double(x0); %#ok<AGROW> % template amplitudes
        st3(irange,4) = double(vexp); %#ok<AGROW> % residual variance of this spike
        st3(irange,5) = korder; %#ok<AGROW> % batch from which this spike was found

        fW(:, irange) = gather(featW); %#ok<AGROW> % template features for this batch
        fWpc(:, :, irange) = gather(featPC); %#ok<AGROW> % PC features

        ntot = ntot + numel(x0); % keeps track of total number of spikes so far
    end

    if ibatch == iter_finalize
        % first iteration of final extraction pass
        % allocate variables when switching to extraction phase
        st3 = zeros(1e7, 5);
%         rez.dWU0A = zeros(nt0, Nchan, Nfilt, nBatches, 'single'); % spike average by template, by batch
%         rez.dWUA = zeros(nt0, Nchan, Nfilt, nBatches, 'single'); % spike average by template, by batch

        % these next three store the low-d template decompositions
        rez.WA = zeros(nt0, Nfilt, Nrank,nBatches,  'single');
        rez.UA = zeros(Nchan, Nfilt, Nrank,nBatches,  'single');
        rez.muA = zeros(Nfilt, nBatches,  'single');

        % these ones store features per spike
        fW  = zeros(Nnearest, 1e7, 'single'); % Nnearest is the number of nearest templates to store features for
        fWpc = zeros(NchanNear, Nrank, 1e7, 'single'); % NchanNear is the number of nearest channels to take PC features from
    end

    prog.update(ibatch);

    if rem(ibatch, 100)==1 || ibatch == iter_finalize
        prog.pause_for_output();
        if ibatch < iter_finalize
            state = 'Initial pass';
        else
            state = 'Extraction pass';
        end
        fprintf('%s: %2.2f sec, %d / %d iterations, %d units, nspks: %2.4f, mu: %2.4f, nst0: %d, merges: %2.4f, %2.4f \n', ...
            state, toc, ibatch, niter, Nfilt, sum(nsp), median(mu), numel(st0), ndrop)

        if ops.fig
            if ibatch==1
                figHand = figure;
            elseif ops.fig
                figure(figHand);
            end
           subplot(2,2,1)
           imagesc(W(:,:,1))
           title('Temporal Components')
           xlabel('Unit number');
           ylabel('Time (samples)');

           subplot(2,2,2)
           imagesc(U(:,:,1))
           title('Spatial Components')
           xlabel('Unit number');
           ylabel('Channel number');

           subplot(2,2,3)
           plot(mu)
           ylim([0 100])
           title('Unit Amplitudes')
           xlabel('Unit number');
           ylabel('Amplitude (arb. units)');

           subplot(2,2,4)
           semilogx(1+nsp, mu, '.')
           ylim([0 100])
           xlim([0 100])
           title('Amplitude vs. Spike Count')
           xlabel('Spike Count');
           ylabel('Amplitude (arb. units)');
           drawnow
        end
    end
end
prog.finish();

if ~ops.useRAM
    fclose(fid);
end

toc

% discards the unused portion of the arrays
st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

if reproducible
    [st3, sortIdx] = sortrows(st3);
    fW = fW(:, sortIdx);
    fWpc = fWpc(:, :, sortIdx);
end

rez.st3 = st3;
rez.st2 = st3; % keep also an st2 copy, because st3 will be over-written by one of the post-processing steps

% the similarity score between templates is simply the correlation,
% taken as the max over several consecutive time delays
rez.simScore = gather(max(WtW, [], 3));

% the template features are stored in cProj, like in Kilosort1
rez.cProj    = fW';
% the neihboring templates idnices are stored in iNeigh
rez.iNeigh   = gather(iList);

rez.ops = ops; % update these (only rez comes out of this script)
rez.nsp = nsp;

%  permute the PC projections in the right order
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');
% iNeighPC keeps the indices of the channels corresponding to the PC features
rez.iNeighPC    = gather(iC(:, iW));

% this whole next block is just done to compress the compressed templates
% we separately svd the time components of each template, and the spatial components
% this also requires a careful decompression function, available somewhere in the GUI code
nKeep = min(Nchan*3,20); % how many PCs to keep
rez.W_a = zeros(nt0 * Nrank, nKeep, Nfilt, 'single');
rez.W_b = zeros(nBatches, nKeep, Nfilt, 'single');
rez.U_a = zeros(Nchan* Nrank, nKeep, Nfilt, 'single');
rez.U_b = zeros(nBatches, nKeep, Nfilt, 'single');
for j = 1:Nfilt
    % do this for every template separately
    WA = reshape(rez.WA(:, j, :, :), [], nBatches);
    WA = gpuArray(WA); % svd on the GPU was faster for this, but the Python randomized CPU version might be faster still
    [A, B, C] = svdecon(WA);
    % W_a times W_b results in a reconstruction of the time components
    rez.W_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.W_b(:,:,j) = gather(C(:, 1:nKeep));

    UA = reshape(rez.UA(:, j, :, :), [], nBatches);
    UA = gpuArray(UA);
    [A, B, C] = svdecon(UA);
    % U_a times U_b results in a reconstruction of the time components
    rez.U_a(:,:,j) = gather(A(:, 1:nKeep) * B(1:nKeep, 1:nKeep));
    rez.U_b(:,:,j) = gather(C(:, 1:nKeep));
end

rez.iW = iW;

% initialize the column labels for merge and split
rez.st3_template_col = 2;
rez.st3_cluster_col = 2;

rez.dWU = gather(rez.dWU);
rez.nsp = gather(rez.nsp);
rez.iW_preSplit = gather(rez.iW);  % needed for reextract

fprintf('Finished compressing time-varying templates \n')
%%
