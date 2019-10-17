function rez = reextractSpikesWithAppliedMask(ks, ops, replace, extractTimeMask)
% the goal here is to re-do the spike extraction phase of Kilosort2 for specific windows of time, where we keep all the 
% templates fixed as they were when KS2 was run on the file, but overwriting the data in these specific windows either 
% with different data or by zeroing certain samples with a mask.  

% replace is a nChSorted x nTime sparse matrix of data values that will overwrite the data in the raw imec file
% extractTimeMask is a nTime length sparse logical indicating during which times spikes extracted should be returned

rez = preprocessDataSub(rez);

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
Nfilt 	= ops.Nfilt;
Nchan 	= ops.Nchan;

rez = struct();
rez.xc = ks.channel_positions_sorted(:, 1);
rez.yc = ks.channel_positions_sorted(:, 2);
[iC, mask, C2C] = getClosestChannels(rez, sigmaMask, NchanNear);
clear rez;

% isortbatches = ks.batch_sort_order; % rez.iorig(:)
% nhalf = ceil(nBatches/2);

% % modified schedule from original KS2 to do a full pass through the data
% modified_schedule = getOr(ops, 'use_modified_schedule', true);
% if modified_schedule
%     ischedule = [1:nBatches nBatches:-1:nhalf];
%     i1 = (nhalf-1):-1:1;
%     i2 = nhalf:nBatches;
%     
%     iter_revert_middle = numel(ischedule) + numel(i1) + 1; % halfway through final pass, revert back to the memorized W for middle batch
%     iter_finalize = numel(ischedule); % finalize and memorizeW
    iter_final_pm = nBatches; % finalize annealing parameter pm using final momentum 
% else
%     ischedule = [nhalf:nBatches nBatches:-1:nhalf];
%     i1 = (nhalf-1):-1:1;
%     i2 = nhalf:nBatches;
% 
%     iter_revert_middle = nBatches + nhalf + 1; % should be >iter-nBatches where korder ==  nhalf
%     iter_finalize = nBatches + 1; % should match niter-nBatches
%     iter_final_pm = nBatches + 1; % should match niter-nBatches
% end
%     
% irounds = cat(2, ischedule, i1, i2); % this is the way we traverse the sorted-batch-list, second half, back to center

% niter   = numel(irounds);
% if irounds(niter - nBatches)~=nhalf
%     error('mismatch between number of batches');
% end
% flag_final = 0;
% flag_resort      = 1;

t0 = ceil(ops.trange(1) * ops.fs);

nInnerIter  = 60;
iter_final_pm = nBatches; % clamped from above
pmi = exp(-1./linspace(ops.momentum(1), ops.momentum(2), iter_final_pm));

Nsum = 7; % how many channels to extend out the waveform in mexgetspikes
Params     = double([NT Nfilt ops.Th(1) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi(1) Nchan NchanNear ops.nt0min 1 Nsum NrankPC ops.Th(1)]);

% W0 = permute(double(wPCA), [1 3 2]);

iList = int32(gpuArray(zeros(Nnearest, Nfilt)));

nsp = gpuArray.zeros(0,1, 'double');

Params(13) = 0;

[Ka, Kb] = getKernels(ops, 10, 1);

p1 = .95; % decay of nsp estimate

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

if ~ops.useRAM
    fid = fopen(ops.fproc, 'r');
end

ntot = 0;
ndrop = zeros(1,2);

m0 = ops.minFR * ops.NT/ops.fs;

prog = ProgressBar(niter, 'Re-extracting modified batches');
for ibatch = 1:nBatches
    k = ibatch;
%     %if ibatch>niter-nBatches && korder==nhalf
%     if ibatch == iter_revert_middle
%         % halfway through in the final pass, we've reached the end of the 
%         [W, dWU] = revertW(rez);
%         prog.pause_for_output();
%         fprintf('Reverted back to middle batch templates\n')
%     end

%     %if ibatch<=niter-nBatches
%     if ibatch <= iter_final_pm 
%         % in initial pass, load up annealing schedule, 
%         % the value of pmi(end) will be used for the final extraction pass
%         Params(9) = pmi(ibatch);
%         pm = pmi(ibatch) * gpuArray.ones(Nfilt, 1, 'double');
%     else
%         pm = pmi(end) * gpuArray.ones(Nfilt, 1, 'double');
%     end

    % dat load \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    if ~ops.useRAM
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
        [dWU, cmap] = mexGetSpikes2(Params, dataRAW, wTEMP, iC-1);         %#ok<ASGLU>
%         dWU = mexGetSpikes(Params, dataRAW, wPCA);
        dWU = double(dWU);
        dWU = reshape(wPCAd * (wPCAd' * dWU(:,:)), size(dWU));
        
        
        W = W0(:,ones(1,size(dWU,3)),:);
        Nfilt = size(W,2);
        nsp(1:Nfilt) = m0;        
        Params(2) = Nfilt;
    end
    
    if flag_resort
        % initial pass, resort the templates by amplitude
        [~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
        iW = int32(squeeze(iW));

        [iW, isort] = sort(iW);
        W = W(:,isort, :);
        dWU = dWU(:,:,isort);
        nsp = nsp(isort);        
    end

    % decompose dWU by svd of time and space (61 by 61)
    [W, U, mu] = mexSVDsmall2(Params, dWU, W, iC-1, iW-1, Ka, Kb);
    
    W = ks.W_batch(:, :, :, k);
    U = ks.U_batch(:, :, :, k);
  
    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); %#ok<ASGLU>
   
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp] = ...
        mexMPnu8(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA);
   
    fexp = exp(double(nsp0).*log(pm(1:Nfilt)));
    fexp = reshape(fexp, 1,1,[]);
    nsp = nsp * p1 + (1-p1) * double(nsp0);
    dWU = dWU .* fexp + (1-fexp) .* (dWU0./reshape(max(1, double(nsp0)), 1,1, []));
    
    % \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
    if ibatch==iter_finalize
        prog.update(ibatch, 'Performing final extraction pass over batches');
        % halfway done, so we switch to final extraction mode and memorizeW
        
    end

    if ibatch< iter_finalize %-50
        % initial pass through the batches
        if rem(ibatch, 5)==1
            % this drops templates
            [W, U, dWU, mu, nsp, ndrop] = ...
                triageTemplates2(ops, iW, C2C, W, U, dWU, mu, nsp, ndrop);
        end
        Nfilt = size(W,2);
        Params(2) = Nfilt;

        % this adds templates        
%         dWU0 = mexGetSpikes(Params, drez, wPCA);
        [dWU0,cmap] = mexGetSpikes2(Params, drez, wTEMP, iC-1); %#ok<ASGLU>
        
        if size(dWU0,3)>0    
            dWU0 = double(dWU0);
            dWU0 = reshape(wPCAd * (wPCAd' * dWU0(:,:)), size(dWU0));            
            dWU = cat(3, dWU, dWU0);

            W(:,Nfilt + (1:size(dWU0,3)),:) = W0(:,ones(1,size(dWU0,3)),:);

            nsp(Nfilt + (1:size(dWU0,3))) = ops.minFR * NT/ops.fs;
            mu(Nfilt + (1:size(dWU0,3)))  = 10;            

            Nfilt = min(ops.Nfilt, size(W,2));
            Params(2) = Nfilt;

            W   = W(:, 1:Nfilt, :);
            dWU = dWU(:, :, 1:Nfilt);
            nsp = nsp(1:Nfilt);
            mu  = mu(1:Nfilt);            
        end

    end

    if ibatch>iter_finalize
        % in the final extraction pass, not the first iteration though
%         rez.dWUA(:,:,:,k) = gather(dWU);
%         rez.WA(:,:,:,k) = gather(W);
%         rez.UA(:,:,:,k) = gather(U);
%         rez.muA(:,k) = gather(mu);
        
        ioffset         = ops.ntbuff;
        if k==1
            ioffset         = 0;
        end
        toff = nt0min + t0 -ioffset + (NT-ops.ntbuff)*(k-1);        
        
        st = toff + double(st0);
        irange = ntot + (1:numel(x0));
        
        if ntot+numel(x0)>size(st3,1)
           fW(:, 2*size(st3,1))    = 0;  %#ok<AGROW>
           fWpc(:,:,2*size(st3,1)) = 0;  %#ok<AGROW>
           st3(2*size(st3,1), 1)   = 0;  %#ok<AGROW>
        end
        
        st3(irange,1) = double(st); %#ok<AGROW>
        st3(irange,2) = double(id0+1); %#ok<AGROW>
        st3(irange,3) = double(x0);         %#ok<AGROW>
        st3(irange,4) = double(vexp); %#ok<AGROW>
        st3(irange,5) = korder; %#ok<AGROW>

        fW(:, irange) = gather(featW);  %#ok<AGROW>

        fWpc(:, :, irange) = gather(featPC);  %#ok<AGROW>

        ntot = ntot + numel(x0);
    end

    if ibatch==iter_finalize 
        % first iteration of final extraction pass
        st3 = zeros(1e7, 5);
        rez.dWUA = zeros(nt0, Nchan, Nfilt, nBatches, 'single'); % spike average by template, by batch
        rez.WA = zeros(nt0, Nfilt, Nrank,nBatches,  'single');
        rez.UA = zeros(Nchan, Nfilt, Nrank,nBatches,  'single');
        rez.muA = zeros(Nfilt, nBatches,  'single');
        
        fW  = zeros(Nnearest, 1e7, 'single');
        fWpc = zeros(NchanNear, Nrank, 1e7, 'single');
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


st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

% ntot;

% [~, isort] = sort(st3(:,1), 'ascend');
% fW = fW(:, isort);
% fWpc = fWpc(:,:,isort);
% st3 = st3(isort, :);

rez.st3 = st3;
rez.st2 = st3;

rez.simScore = gather(max(WtW, [], 3));

rez.cProj    = fW';
rez.iNeigh   = gather(iList);

rez.ops = ops;

rez.nsp = nsp;

% nNeighPC        = size(fWpc,1);
rez.cProjPC     = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

% [~, iNch]       = sort(abs(rez.U(:,:,1)), 1, 'descend');
% maskPC          = zeros(Nchan, Nfilt, 'single');
rez.iNeighPC    = gather(iC(:, iW));


nKeep = 20; % how many PCs to keep
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

fprintf('Finished compressing time-varying templates \n')
%%



