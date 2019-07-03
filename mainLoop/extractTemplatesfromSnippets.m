function [wTEMP, wPCA] = extractTemplatesfromSnippets(rez, nPCs, use_distrusted)

ops = rez.ops;

nskip = getOr(ops, 'nskip', 25);
% Nchan 	= ops.Nchan;

Nbatch      = rez.temp.Nbatch;

NT  	= ops.NT;
batchstart = 0:NT:NT*Nbatch;

% extract the PCA projections
if ~ops.useRAM
    fid = fopen(ops.fproc, 'r');
end

k = 0;
dd = gpuArray.zeros(ops.nt0, 5e4, 'single');
prog = ProgressBar(Nbatch, 'Extracting initial template snippets');
for ibatch = 1:nskip:Nbatch
    if ~ops.useRAM
        offset = 2 * ops.Nchan*batchstart(ibatch);
        fseek(fid, offset, 'bof');
        dat = fread(fid, [NT ops.Nchan], '*int16');
    else
        dat = rez.DATA(:, :, ibatch);
    end
    
    % don't remove distrusted samples, just ignore spikes that live there
    if ~use_distrusted && ~isempty(rez.distrust_batched)
        distrust_this_batch = rez.distrust_batched(:, ibatch);
    else
        distrust_this_batch = [];
    end
    
    % move data to GPU and scale it
    if ops.GPU
        dataRAW = gpuArray(dat);
    else
        dataRAW = dat;
    end
    dataRAW = single(dataRAW);
    dataRAW = dataRAW / ops.scaleproc;
    
    
    % find isolated spikes (ignoring distrusted)
    [row, col, mu] = isolated_peaks_new(dataRAW, ops, distrust_this_batch);
    
    clips = get_SpikeSample(dataRAW, row, col, ops, 0);
    
    c = sq(clips(:, :));
        
    if k+size(c,2)>size(dd,2)
        dd(:, 2*size(dd,2)) = 0;
    end
    
    prog.update(ibatch);
    
    dd(:, k + [1:size(c,2)]) = c;    
    k = k + size(c,2);
    if k>1e5
        break;
    end
end
if ~ops.useRAM
    fclose(fid);
end
prog.finish();

dd = dd(:, 1:k);

wTEMP = dd(:, randperm(size(dd,2), nPCs));
wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5;

for i = 1:10
   cc = wTEMP' * dd;
   [amax, imax] = max(cc,[],1);
   for j = 1:nPCs
      wTEMP(:,j)  = dd(:,imax==j) * amax(imax==j)';
   end
   wTEMP = wTEMP ./ sum(wTEMP.^2,1).^.5;
end

dd = double(gather(dd));
[U Sv V] = svdecon(dd);

wPCA = gpuArray(single(U(:, 1:nPCs)));
wPCA(:,1) = - wPCA(:,1) * sign(wPCA(ops.nt0min,1));
