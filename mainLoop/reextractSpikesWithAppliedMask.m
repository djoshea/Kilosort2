function out = reextractSpikesWithAppliedMask(ks, varargin)
% the goal here is to re-do the spike extraction phase of Kilosort2 for specific windows of time, where we keep all the 
% templates fixed as they were when KS2 was run on the file, but overwriting the data in these specific windows either 
% with different data or by zeroing certain samples with a mask.  

% replace is a nChSorted x nTime sparse matrix of data values that will overwrite the data in the raw imec file
% extractTimeMask is a nTime length sparse logical indicating during which times spikes extracted should be returned

% this will load back in our data to RAM (if ops.useRAM) or re-write it to the ops.fproc file

p = inputParser();
p.addParameter('rez_pre_loaded', [], @(x) isempty(x) || isstruct(x));
p.addParameter('batch_inds', [], @(x) true);
p.parse(varargin{:});

assert(~isempty(ks.W));
assert(~isempty(ks.U));
assert(~isempty(ks.mu));
assert(~isempty(ks.W_batch));
assert(~isempty(ks.U_batch));
assert(~isempty(ks.mu_batch));

ops = ks.ops;
ops.chanMap = fullfile(ops.root,'chanMap.mat');
rez = p.Results.rez_pre_loaded;
if ~isempty(rez)
    rez.ops = ops;
else
    rez = preprocessDataSub(ops);
    assignin('base', 'rez_re', rez);
    ops = rez.ops;
end

batch_inds = p.Results.batch_inds;
if islogical(batch_inds)
    assert(numel(batch_inds) == ks.nBatches);
    batch_inds = find(batch_inds);
end

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

Nfilt_preSplit = ks.nTemplatesPreSplit;

Nsum = 7; % how many channels to extend out the waveform in mexgetspikes
Params     = double([NT Nfilt ops.Th(end) nInnerIter nt0 Nnearest ...
    Nrank ops.lam pmi_end Nchan NchanNear ops.nt0min 2 Nsum NrankPC ops.Th(1)]);
% Params(3) set to ops.Th(end) --> different threshold on extraction pass
% Params(9) set to pmi(end) == exp(-1/ops.momentum(2))
% Params(13) set to 2  --> extract ALL features on the extraction pass

fprintf('Time %3.0fs. Optimizing templates ...\n', toc)

if ~ops.useRAM
    fid = fopen(ops.fproc, 'r');
end

% normally KS2 would extract dWU from the data, then decompose into W, U, and mu
% here we reconstruct approximate dWU from W, U, and mu
W = ks.W;
U = ks.U;
mu = ks.mu;
dWU = zeros(nt0, Nchan, Nfilt, 'gpuArray');
for n = 1:Nfilt
    dWU(:, :, n) = mu(n) * sq(W(:,n,:)) * sq(U(:,n,:))';
end
[~, iW] = max(abs(dWU(nt0min, :, :)), [], 2);
iW = int32(squeeze(iW));

[WtW, iList] = getMeWtW(single(W), single(U), Nnearest); %#ok<ASGLU>
wPCA = ops.wPCA;

% first iteration of final extraction pass
prog = ProgressBar(nBatches, 'Re-extracting modified batches');

ntot = 0;
st3 = zeros(1e7, 5);
fW  = zeros(Nnearest, 1e7, 'single');
fWpc = zeros(NchanNear, Nrank, 1e7, 'single');

for k = 1:nBatches
    if ~ismember(k, batch_inds)
        continue;
    end
    
    if ~ops.useRAM
        offset = 2 * ops.Nchan*batchstart(k);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
        dat = rez.DATA(:, :, k);
    end
    dataRAW = single(gpuArray(dat))/ ops.scaleproc;

    Params(1) = size(dataRAW, 1); % update NT each loop in case we subset dataRAW
    
    W = ks.W_batch(:, :, :, k);
    U = ks.U_batch(:, :, :, k);
    mu = ks.mu_batch(:, k);
    
    % this needs to change
    [UtU, maskU] = getMeUtU(iW, iC, mask, Nnearest, Nchan); %#ok<ASGLU>
   
    [st0, id0, x0, featW, dWU0, drez, nsp0, featPC, vexp] = ...
        mexMPnu8r(Params, dataRAW, single(U), single(W), single(mu), iC-1, iW-1, UtU, iList-1, ...
        wPCA); %#ok<ASGLU>
   
    ioffset         = ops.ntbuff;
    if k==1
        ioffset         = 0;
    end
    toff = nt0min + t0 -ioffset + (NT-ops.ntbuff)*(k-1);        
        
    st = toff + double(st0);
    irange = ntot + (1:numel(x0));
        
    if ntot+numel(x0)>size(st3,1)
       fW(:, 2*size(st3,1))    = 0;
       fWpc(:,:,2*size(st3,1)) = 0;
       st3(2*size(st3,1), 1)   = 0;
    end
        
    st3(irange,1) = double(st);
    st3(irange,2) = double(id0+1);
    st3(irange,3) = double(x0);
    st3(irange,4) = double(vexp);
    st3(irange,5) = k;

    fW(:, irange) = gather(featW);
    fWpc(:, :, irange) = gather(featPC);

    ntot = ntot + numel(x0);
    prog.update(k);
end
prog.finish();

if ~ops.useRAM
    fclose(fid);
end

st3 = st3(1:ntot, :);
fW = fW(:, 1:ntot);
fWpc = fWpc(:,:, 1:ntot);

out.st3 = st3;
out.cProj = fW';
out.cProjPC = permute(fWpc, [3 2 1]); %zeros(size(st3,1), 3, nNeighPC, 'single');

% need to apply thresholds from set_cutoff 
% vexp = rez.st3(ix,4);

fprintf('Finished compressing time-varying templates \n');



