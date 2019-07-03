function wPCA = extractPCfromSnippets(rez, nPCs)

ops = rez.ops;

% Nchan 	= ops.Nchan;

Nbatch      = rez.temp.Nbatch;

NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

% extract the PCA projections
CC = zeros(ops.nt0);

if ~rez.ops.useRAM
    fid = fopen(ops.fproc, 'r');
end

for ibatch = 1:100:Nbatch
    if ~rez.ops.useRAM
        offset = 2 * ops.Nchan*batchstart(ibatch);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
        dat = rez.DATA(:, :, ibatch);
    end
    
    if ~isempty(rez.distrust_batched)
        distrust_this_batch = rez.distrust_batched(:, ibatch);
        dat = dat(~distrust_this_batch, :);
    end
    
    % move data to GPU and scale it
    if ops.GPU
        dataRAW = gpuArray(dat);
    else
        dataRAW = dat;
    end
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    % find isolated spikes
    [row, col, mu] = isolated_peaks_new(dataRAW, ops);
    
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);
    
    c = sq(clips(:, :));
    CC = CC + gather(c * c')/1e3;
    
end
if ~rez.ops.useRAM
    fclose(fid);
end

[U Sv V] = svdecon(CC);

wPCA = U(:, 1:nPCs);

wPCA(:,1) = - wPCA(:,1) * sign(wPCA(21,1));
