function exportRezToMat(rez, fname)

% drop features from rez, too large
rez = clearFields(rez, {'temp', 'cProj', 'cProjPC', 'cProj_cutoff_invalid', 'cProjPC_cutoff_invalid', 'dWUA'});
if isfield(rez, 'ops') && isfield(rez.ops, 'gui')
    rez.ops.gui = [];
end

% sort spike times
[~, isort]   = sort(rez.st3(:,1), 'ascend');
rez.st3      = rez.st3(isort, :);

% gather all gpuArrays
flds = fieldnames(rez);
for iF = 1:numel(flds)
    val = rez.(flds{iF});
    if isa(val, 'gpuArray')
        rez.(flds{iF}) = gather(val);
    end
end

% save final results as rez
save(fname, 'rez', '-v7.3');

end

function s = clearFields(s, flds)
    for iF = 1:numel(flds)
        fld = flds{iF};
        if isfield(s, fld)
            s = rmfield(s, fld);
        end
    end
end  