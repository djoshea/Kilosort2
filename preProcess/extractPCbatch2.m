function [uS, idchan, st] = extractPCbatch2(rez, wPCA, ibatch, iC)
% this function finds threshold crossings in the data using
% projections onto the pre-determined principal components
% wPCA is number of time samples by number of PCs
% ibatch is a scalar indicating which batch to analyze
% iC is NchanNear by Nchan, indicating for each channel the nearest
% channels to it

% starts with predefined PCA waveforms
wPCA = gpuArray(single(wPCA(:, 1:3)));
ops = rez.ops;
reproducible = getOr(ops, 'reproducible', false);
Nbatch      = rez.temp.Nbatch;
NT  	      = ops.NT;

% number of nearest channels to consider
NchanNear = size(iC,1);

if ~ops.useRAM
    fid = fopen(ops.fproc, 'r');
    batchstart = 0:NT:NT*Nbatch; % batches start at these timepoints
    offset = 2 * ops.Nchan*batchstart(ibatch); % binary file offset in bytes
    fseek(fid, offset, 'bof');
    dat = fread(fid, [NT ops.Nchan], '*int16');
    fclose(fid);
else
    dat = rez.DATA(:, :, ibatch);
end

if ~isempty(rez.distrust_batched)
    distrust_this_batch = rez.distrust_batched(:, ibatch);
    dat = dat(~distrust_this_batch, :);
end

% move data to GPU and scale it
dataRAW = gpuArray(dat);
dataRAW = single(dataRAW);
dataRAW = dataRAW / ops.scaleproc;

nt0min = rez.ops.nt0min;
spkTh = ops.ThPre;
[nt0, NrankPC] = size(wPCA);
[NT, Nchan] = size(dataRAW);

% another Params variable to take all our parameters into the C++ code
Params = [NT Nchan NchanNear nt0 nt0min spkTh NrankPC];

% call a CUDA function to do the hard work
% returns a matrix of features uS, as well as the center channels for each spike
if reproducible
    % output is not in reproducible order, sorting by time then channel fixes this
    [uS, idchan, st] = mexThSpkPCr(Params, dataRAW, wPCA, iC-1);
    [out, sortIdx] = sortrows([st, idchan]);
    st = out(:, 1);
    idchan = out(:, 2) + 1; % go from 0-indexing to 1-indexing
    uS = uS(:, sortIdx);
else
    [uS, idchan] = mexThSpkPC(Params, dataRAW, wPCA, iC-1);
    idchan = idchan + 1;
    st = [];
end
