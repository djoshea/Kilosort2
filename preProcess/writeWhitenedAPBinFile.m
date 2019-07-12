function writeWhitenedAPBinFile(rez, fname)

%assert(isfield(rez, 'DATA') && ~isempty(rez.DATA), 'Requires rez.DATA field, loaded when ops.useRAM==true');
ops = rez.ops;

prog = ProgressBar(ops.Nbatch, 'Writing whitened batches to %s', fname);

chanMap = ops.chanMap;

syncCh = ops.NchanTOT;
blank_ch_inds = setdiff((1:ops.NchanTOT)', [syncCh;  chanMap]);

NchanTOT = ops.NchanTOT;
NT = ops.NT;

% we need to read from the original binary file in order to get the raw data to store the modified trials into
fidRaw         = fopen(ops.fbinary, 'r');
if ~ops.useRAM
    fidWhitened        = fopen(ops.fproc,   'r');
end

fidDestination = fopen(fname, 'w');

for ibatch = 1:ops.Nbatch
    
    % LOAD RAW DATA FROM ORIGINAL FILE
    % skip the ntbuff, we're not filtering anymore, just taking the raw data so we get the non-whitened channels
    offset = max(0, ops.twind + 2*NchanTOT*(NT * (ibatch-1)));
    fseek(fidRaw, offset, 'bof');
    
    dataRAW = fread(fidRaw, [NchanTOT NT], '*int16');
    if isempty(dataRAW)
        break;
    end
    nsampcurr = size(dataRAW,2);
    
    % LOAD FILTERED DATA FROM DISK OR RAM
    if ~ops.useRAM
        offset = 2 * ops.Nchan*NT*(iBatch-1);
        fseek(fid, offset, 'bof');
        W_dat = fread(fidWhitened, [NT ops.Nchan], '*int16')'; % --> C x T (written as T x C)
    else
        W_dat = rez.DATA(:, :, ibatch)'; % --> C x T
    end
    
    % OVERWRITE chanmap'ed channels in dataRaw with whitened data
    if nsampcurr<NT
        W_dat = W_dat(:, 1:nsampcurr);
    end
    
    dataRAW(chanMap, :) = W_dat;
    
    dataRAW(blank_ch_inds, :) = 0; % blank the other non-sync channels
    
    fwrite(fidDestination, dataRAW, 'int16');
    
    prog.update(ibatch);
end
prog.finish();

if ~ops.useRAM
    fclose(fidWhitened);
end
fclose(fidRaw);
fclose(fidDesintation);

end
