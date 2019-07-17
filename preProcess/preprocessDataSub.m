function [rez, DATA] = preprocessDataSub(ops)
tic;
ops.nt0 	= getOr(ops, {'nt0'}, 61);
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61));

NT       = ops.NT ;
NchanTOT = ops.NchanTOT;

bytes = get_file_size(ops.fbinary);
nTimepoints = floor(bytes/NchanTOT/2);
ops.tstart = ceil(ops.trange(1) * ops.fs);
ops.tend   = min(nTimepoints, ceil(ops.trange(2) * ops.fs));
ops.sampsToRead = ops.tend-ops.tstart;
ops.twind = ops.tstart * NchanTOT*2;

Nbatch      = ceil(ops.sampsToRead /(NT-ops.ntbuff));
ops.Nbatch = Nbatch;

[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap);
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault);

if getOr(ops, 'minfr_goodchannels', .1)>0
    
    % determine bad channels
    fprintf('Time %3.0fs. Determining good channels.. \n', toc);

    igood = get_good_channels(ops, chanMap, rez.DATA); % trusted samples only
    xc = xc(igood);
    yc = yc(igood);
    kcoords = kcoords(igood);
    chanMap = chanMap(igood);
        
    ops.igood = igood;
else
    ops.igood = true(size(chanMap));
end

ops.Nchan = numel(chanMap);
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan;

rez.ops         = ops;
rez.xc = xc;
rez.yc = yc;

rez.xcoords = xc;
rez.ycoords = yc;

% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords; 


NTbuff      = NT + 4*ops.ntbuff;


% by how many bytes to offset all the batches
rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;


fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

% this requires removing bad channels first
Wrot = get_whitening_matrix(rez); % trusted samples only


fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');
if ~ops.useRAM
    fidW        = fopen(ops.fproc,   'w');
    DATA = [];
else
    gbData = NT * rez.ops.Nchan * Nbatch * 2 / 2^30;
    fprintf('Allocating %.2f GiB of data in RAM, this make take some time\n', gbData);
    DATA = zeros(NT, rez.ops.Nchan, Nbatch, 'int16');
end
% load data into patches, filter, compute covariance
if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
    [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
else
    [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
end

distrust_data_mask = getOr(ops, 'distrust_data_mask', []);
if isempty(distrust_data_mask)
    distrust_batched = [];
else
    distrust_batched = true(NT, Nbatch);
end

% in each loop, we start at 2*ntbuff before the current batch start (ibatch-1)*NT-ntbuff
% NTbuff = NT + 4*ntbuff samples are read (2*ntbuff before, and 2*ntbuff after)
% then NT are kept, skipping the first ntbuff samples (except for the first batch). 
% This means that the left edge of each batched data block has ntbuff overlapping samples at the left (except for the first batch)
% This is factored in when detecting spike times in learnAndSolve8b (see calculation of toff)
prog = ProgressBar(Nbatch, 'Preprocessing batches');
for ibatch = 1:Nbatch
    offset = max(0, ops.twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    if offset==0
        ioffset = 0;
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof');
    
    buff = fread(fid, [NchanTOT NTbuff], '*int16');
    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end
    
    if ops.GPU
        dataRAW = gpuArray(buff);
    else
        dataRAW = buff;
    end
    dataRAW = dataRAW';
    dataRAW = single(dataRAW);
    dataRAW = dataRAW(:, chanMap);
    
    % only select trusted timepoints for mean computation
    if ~isempty(distrust_data_mask)
        inds_this_batch = max(0, ops.tstart + (NT-ops.ntbuff)*(ibatch-1)-ops.ntbuff) + (1 : size(dataRAW, 1));
        inds_this_batch = inds_this_batch(inds_this_batch <= numel(distrust_data_mask));
        distrust_this_batch = distrust_data_mask(inds_this_batch);
        dataRAW_trusted = dataRAW(~distrust_this_batch, :);
    else
        dataRAW_trusted = dataRAW;
    end
    
    % subtract the mean from each channel
    dataRAW = dataRAW - mean(dataRAW_trusted, 1);    
    
    datr = filter(b1, a1, dataRAW);
    datr = flipud(datr);
    datr = filter(b1, a1, datr);
    datr = flipud(datr);
    
    % CAR, common average referencing by median
    if getOr(ops, 'CAR', 1)
        datr = datr - median(datr, 2);
    end
    
    datr = datr(ioffset + (1:NT),:);
    inds_keep = ioffset + (1:NT);
    inds_keep = inds_keep(inds_keep < numel(distrust_this_batch));
    distrust_batched(1:numel(inds_keep), ibatch) = distrust_this_batch(inds_keep); %#ok<AGROW>
    
    datr    = datr * Wrot;
    
    if ops.useRAM
        DATA(:,:,ibatch) = gather_try(datr); %#ok<AGROW>
    else
        datcpu  = gather_try(int16(datr));
        fwrite(fidW, datcpu, 'int16');
    end
    prog.update(ibatch);
end
prog.finish();

Wrot        = gather_try(Wrot);
rez.Wrot    = Wrot;

if ~ops.useRAM
    fclose(fidW);
end
fclose(fid);

fprintf('Time %3.0fs. Finished preprocessing %d batches. \n', toc, Nbatch);

rez.temp.Nbatch = Nbatch;
rez.DATA = DATA;
rez.distrust_batched = distrust_batched;
