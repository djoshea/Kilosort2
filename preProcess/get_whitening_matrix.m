function Wrot = get_whitening_matrix(rez)
% based on a subset of the data, compute a channel whitening matrix
% this requires temporal filtering first (gpufilter)

ops = rez.ops;
Nbatch = ops.Nbatch;
twind = ops.twind;
NchanTOT = ops.NchanTOT;
NT = ops.NT;
NTbuff = ops.NTbuff;
chanMap = ops.chanMap;
Nchan = rez.ops.Nchan;
xc = rez.xc;
yc = rez.yc;

distrust_data_mask = getOr(ops, 'distrust_data_mask', []);

% load data into patches, filter, compute covariance
do_hp_filter = getOr(ops, 'do_hp_filter', true);
if do_hp_filter
    if isfield(ops,'fslow')&&ops.fslow<ops.fs/2
        [b1, a1] = butter(3, [ops.fshigh/ops.fs,ops.fslow/ops.fs]*2, 'bandpass');
    else
        [b1, a1] = butter(3, ops.fshigh/ops.fs*2, 'high');
    end
end

fprintf('Getting channel whitening matrix... \n');
fid = fopen(ops.fbinary, 'r');
CC = gpuArray.zeros( Nchan,  Nchan, 'single'); % we'll estimate the covariance from data batches, then add to this variable


ibatch = 1;
while ibatch<=Nbatch
    offset = max(0, twind + 2*NchanTOT*((NT - ops.ntbuff) * (ibatch-1) - 2*ops.ntbuff));
    fseek(fid, offset, 'bof');
    buff = fread(fid, [NchanTOT NTbuff], '*int16');

    if isempty(buff)
        break;
    end
    nsampcurr = size(buff,2);
    if nsampcurr<NTbuff
        buff(:, nsampcurr+1:NTbuff) = repmat(buff(:,nsampcurr), 1, NTbuff-nsampcurr);
    end

    % only select trusted timepoints
    if ~isempty(distrust_data_mask)
        inds_this_batch = max(0, ops.tstart + (NT-ops.ntbuff)*(ibatch-1)-ops.ntbuff) + (1 : size(buff, 2));
        inds_this_batch = inds_this_batch(inds_this_batch <= numel(distrust_data_mask));
        distrust_this_batch = distrust_data_mask(inds_this_batch);
        buff = buff(:, ~distrust_this_batch);
    end
    NTthis = size(buff, 2);

    datr    = gpufilter(buff, ops, rez.ops.chanMap); % apply filters and median subtraction

    CC        = CC + (datr' * datr)/NTthis; % sample covariance

    ibatch = ibatch + ops.nSkipCov; % skip this many batches
end
CC = CC / ceil((Nbatch-1)/ops.nSkipCov); % normalize by number of batches

fclose(fid);

if ops.whiteningRange<Inf
    % if there are too many channels, a finite whiteningRange is more robust to noise in the estimation of the covariance
    ops.whiteningRange = min(ops.whiteningRange, Nchan);
    Wrot = whiteningLocal(gather(CC), yc, xc, ops.whiteningRange); % this function performs the same matrix inversions as below, just on subsets of channels around each channel
else
    Wrot = whiteningFromCovariance(CC);
end
Wrot    = ops.scaleproc * Wrot; % scale this from unit variance to int 16 range. The default value of 200 should be fine in most (all?) situations.

fprintf('Channel-whitening matrix computed. \n');
