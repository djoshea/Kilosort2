function DATA = loadBinaryDataIntoRAM(ops)

NT = ops.NT;
NchanTOT = ops.NchanTOT;
Nbatch = ops.Nbatch;

bytes = get_file_size(ops.fbinary);
nTimepoints = floor(bytes/NchanTOT/2);

gbData = nTimepoints * ops.Nchan * 2 / 2^30;

mm = memmapfile(ops.fbinary, 'Offset', ops.twind, 'Format', {'int16', [NchanTOT nTimepoints], 'data'});
prog = ProgressBar(Nbatch, 'Loading %.2f GiB of DATA in RAM, this make take some time\n', gbData);
DATA = zeros(NT, NchanTOT, Nbatch, 'int16');    
for iB = 1:Nbatch
    batch_inds = NT*(iB-1) + (1:NT);
    if iB == Nbatch && any(batch_inds > nTimepoints)
        mask = batch_inds <= nTimepoints;
        last_sample_ind = batch_inds(find(mask, 1, 'last'));
        batch_inds = batch_inds(mask);
        DATA(mask, :, iB) = mm.Data.data(:, batch_inds)'; 
        DATA(~mask, :, iB) = repmat(mm.Data.data(:, last_sample_ind)', nnz(~mask), 1);        
    else
        DATA(:, :, iB) = mm.Data.data(:, batch_inds)';
    end
    prog.update(iB);
end
prog.finish();

end