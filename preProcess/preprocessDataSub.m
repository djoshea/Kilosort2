function rez = preprocessDataSub(ops, varargin)
% this function takes an ops struct, which contains all the Kilosort2 settings and file paths
% and creates a new binary file of preprocessed data, logging new variables into rez.
% The following steps are applied:
% 1) conversion to float32;
% 2) common median subtraction (if ops.CAR)
% 3) bandpass filtering (if ops.do_hp_filter)
% 4) channel whitening
% 5) scaling to int16 values

% these are used for reextractingSpikes by selectively overwriting specific windows of time in situ
p = inputParser();
p.addParameter('batch_inds', [], @(x) isempty(x) || isvector(x)); % if specified, only certain batches will be loaded and those batch_inds set in rez.DATA_batch_inds
p.addParameter('data_replace_windows', zeros(0, 2), @(x) ismatrix(x) && size(x, 2) == 2);
p.addParameter('data_replace', {}, @iscell);
p.addParameter('only_batches_overlapping_windows', zeros(0, 2), @(x) ismatrix(x) && size(x, 2) == 2); % only include batches that overlap something in this window
p.parse(varargin{:});

% check data replacement windows
data_replace_windows = uint64(p.Results.data_replace_windows);
data_replace = p.Results.data_replace;
assert(numel(data_replace) == size(data_replace_windows, 1));
for iR = 1:numel(data_replace)
    nTimeExpected = diff(data_replace_windows(iR, [1 2])) + uint64(1);
    nTimeActual = size(data_replace{iR}, 2);
    assert(nTimeExpected == nTimeActual, 'data_replace has %u samples but data_replace_windows indicates %d samples', nTimeActual, nTimeExpected);
end
only_batches_overlapping_windows = p.Results.only_batches_overlapping_windows;

tic;
ops.nt0 	  = getOr(ops, {'nt0'}, 61); % number of time samples for the templates (has to be <=81 due to GPU shared memory)
ops.nt0min  = getOr(ops, 'nt0min', ceil(20 * ops.nt0/61)); % time sample where the negative peak should be aligned

NT       = ops.NT ; % number of timepoints per batch
NchanTOT = ops.NchanTOT; % total number of channels in the raw binary file, including dead, auxiliary etc

bytes       = get_file_size(ops.fbinary); % size in bytes of raw binary
nTimepoints = floor(bytes/NchanTOT/2); % number of total timepoints
ops.tstart  = ceil(ops.trange(1) * ops.fs); % starting timepoint for processing data segment
ops.tend    = min(nTimepoints, ceil(ops.trange(2) * ops.fs)); % ending timepoint
ops.sampsToRead = ops.tend-ops.tstart; % total number of samples to read
ops.twind = ops.tstart * NchanTOT*2; % skip this many bytes at the start

Nbatch      = ceil(ops.sampsToRead /(NT-ops.ntbuff)); % number of data batches
ops.Nbatch = Nbatch;

batch_inds = p.Results.batch_inds;
if isempty(batch_inds)
    batch_inds = (1:ops.Nbatch)';
elseif islogical(batch_inds)
    assert(numel(batch_inds) == ops.Nbatch);
    batch_inds = find(batch_inds);
else
    assert(all(batch_inds > 0 & batch_inds <= ops.Nbatch), 'batch_inds out of range 1:%d', Nbatch);
end

t0 = ceil(ops.trange(1) * ops.fs);
if ~isempty(only_batches_overlapping_windows)
    % subset batches by those that have overlap with data_replace_windows
    mask = false(size(batch_inds));
    for iibatch = 1:numel(batch_inds)
        ibatch = batch_inds(iibatch);
        ioffset         = ops.ntbuff;
        if ibatch==1
            ioffset         = 0;
        end
        tfirst = ops.nt0min + t0 -ioffset + (NT-ops.ntbuff)*(ibatch-1);
        tlast = tfirst + NT - 1;
        mask(iibatch) = any(only_batches_overlapping_windows(:, 1) <= tlast & only_batches_overlapping_windows(:, 2) >= tfirst);
    end
    batch_inds = batch_inds(mask);
end

Nbatch_loaded = numel(batch_inds);
if Nbatch_loaded == 0
    error('No batches specified for loading, check data_replace_windows?');
end
rez.ops.Nbatch_loaded = Nbatch_loaded;

if isvector(ops.chanMap)
    ops.chanMap = fullfile(ops.root,'chanMap.mat');
end
[chanMap, xc, yc, kcoords, NchanTOTdefault] = loadChanMap(ops.chanMap); % function to load channel map file
ops.NchanTOT = getOr(ops, 'NchanTOT', NchanTOTdefault); % if NchanTOT was left empty, then overwrite with the default

if getOr(ops, 'minfr_goodchannels', .1)>0 % discard channels that have very few spikes
    % determine bad channels
    fprintf('Time %3.0fs. Determining good channels.. \n', toc);
    igood = get_good_channels(ops, chanMap);

    chanMap = chanMap(igood); %it's enough to remove bad channels from the channel map, which treats them as if they are dead

    xc = xc(igood); % removes coordinates of bad channels
    yc = yc(igood);
    kcoords = kcoords(igood);

    ops.igood = igood;
else
    ops.igood = true(size(chanMap));
end

ops.Nchan = numel(chanMap); % total number of good channels that we will spike sort
ops.Nfilt = getOr(ops, 'nfilt_factor', 4) * ops.Nchan; % upper bound on the number of templates we can have

rez.ops         = ops; % memorize ops

rez.xc = xc; % for historical reasons, make all these copies of the channel coordinates
rez.yc = yc;
rez.xcoords = xc;
rez.ycoords = yc;
% rez.connected   = connected;
rez.ops.chanMap = chanMap;
rez.ops.kcoords = kcoords;

NTbuff      = NT + 4*ops.ntbuff; % we need buffers on both sides for filtering

rez.ops.Nbatch = Nbatch;
rez.ops.NTbuff = NTbuff;
rez.ops.chanMap = chanMap;

fprintf('Time %3.0fs. Computing whitening matrix.. \n', toc);

% this requires removing bad channels first
% trusted samples only
Wrot = get_whitening_matrix(rez); % outputs a rotation matrix (Nchan by Nchan) which whitens the zero-timelag covariance of the data

fprintf('Time %3.0fs. Loading raw data and applying filters... \n', toc);

fid         = fopen(ops.fbinary, 'r');  % open for reading raw data
if ~ops.useRAM
    fidW        = fopen(ops.fproc,   'w'); % open for writing processed data
    DATA = [];
else
    gbData = NT * rez.ops.Nchan * Nbatch_loaded * 2 / 2^30;
    fprintf('Allocating %.2f GiB of data in RAM, this make take some time\n', gbData);
    DATA = zeros(NT, rez.ops.Nchan, Nbatch_loaded, 'int16');
end

distrust_data_mask = getOr(ops, 'distrust_data_mask', []);
if isempty(distrust_data_mask)
    distrust_batched = [];
else
    distrust_batched = true(NT, Nbatch_loaded); % this starts out as true such that the filter-padding edges aren't trusted either
end

% in each loop, we start at 2*ntbuff before the current batch start (ibatch-1)*NT-ntbuff
% NTbuff = NT + 4*ntbuff samples are read (2*ntbuff before, and 2*ntbuff after)
% then NT are kept, skipping the first ntbuff samples (except for the first batch).
% This means that the left edge of each batched data block has ntbuff overlapping samples at the left (except for the first batch)
% This is factored in when detecting spike times in learnAndSolve8b (see calculation of toff)
prog = ProgressBar(Nbatch_loaded, 'Preprocessing batches');
for iibatch = 1:Nbatch_loaded
    ibatch = batch_inds(iibatch);
    % we'll create a binary file of batches of NT samples, which overlap consecutively on ops.ntbuff samples
    % in addition to that, we'll read another ops.ntbuff samples from before and after, to have as buffers for filtering
    offset = max(0, ops.twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff)); % number of samples to start reading at.
    if offset==0
        ioffset = 0; % The very first batch has no pre-buffer, and has to be treated separately
    else
        ioffset = ops.ntbuff;
    end
    fseek(fid, offset, 'bof'); % fseek to batch start in raw file

    buff = fread(fid, [NchanTOT NTbuff], '*int16'); % read and reshape. Assumes int16 data (which should perhaps change to an option)
    if isempty(buff)
        break; % this shouldn't really happen, unless we counted data batches wrong
    end
    nsampcurr = size(buff,2); % how many time samples the current batch has
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);  % pad with zeros, if this is the last batch
    end

    samples_this_batch = max(0, ops.tstart + (NT-ops.ntbuff)*(ibatch-1)-2*ops.ntbuff) + (1 : NTbuff);
    
    % only select trusted timepoints for mean computation
    if ~isempty(distrust_data_mask)
        inds_this_batch = samples_this_batch;
        inds_this_batch = inds_this_batch(inds_this_batch <= numel(distrust_data_mask));
        distrust_this_batch = distrust_data_mask(inds_this_batch);
    else
        distrust_this_batch = [];
    end
    
    % do data replacment if requested, before subsequent processing
    tfirst = samples_this_batch(1);
    tlast = samples_this_batch(end);
    has_overlap = data_replace_windows(:, 1) <= tlast & data_replace_windows(:, 2) >= tfirst;
    if any(has_overlap)
        for iR = 1:numel(data_replace)
            if ~has_overlap(iR), continue; end
            window = data_replace_windows(iR, :);
            [mask_take, inds_put] = ismember(window(1):window(2), samples_this_batch);
            buff(chanMap, inds_put(mask_take)) = data_replace{iR}(:, mask_take);
        end
    end

    datr = gpufilter(buff, ops, chanMap, distrust_this_batch); % apply filters and median subtraction

    datr = datr(ioffset + (1:NT),:); % remove timepoints used as buffers
    if ~isempty(distrust_data_mask)
        % remove same timepoints from distrust_this_batch
        inds_keep = ioffset + (1:NT);
        inds_keep = inds_keep(inds_keep < numel(distrust_this_batch));
        distrust_batched(1:numel(inds_keep), ibatch) = distrust_this_batch(inds_keep); %#ok<AGROW>
    end
    datr    = datr * Wrot; % whiten the data and scale by 200 for int16 range

    if ops.useRAM
        DATA(:,:,iibatch) = gather_try(int16(datr)); %#ok<AGROW>
    else
        datcpu  = gather_try(int16(datr)); % convert to int16, and gather on the CPU side
        fwrite(fidW, datcpu, 'int16');  % write this batch to binary file
    end
    prog.update(iibatch);
end
prog.finish();

rez.Wrot    = gather(Wrot); % gather the whitening matrix as a CPU variable

if ~ops.useRAM
    fclose(fidW); % close the files
end
fclose(fid);

fprintf('Time %3.0fs. Finished preprocessing %d batches. \n', toc, Nbatch);

rez.temp.Nbatch = Nbatch;
rez.DATA = DATA;
rez.DATA_batch_inds = batch_inds;
rez.distrust_batched = distrust_batched;

