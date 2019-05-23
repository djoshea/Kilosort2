function rez = remergeSplitTemplates(rez)
% splitAllClusters actually splits the template assignment in column 6 of rez.st3
% whereas the cluster identities are in column 7. This goes through and remerges all the split templates
% to share the same cluster id in column 7

split_orig_template = rez.split_orig_template;
current_cluster = rez.st3(:, 7);

Nfilt = size(rez.W,2);
for ik = 1:Nfilt
    if split_orig_template(ik) ~= ik
        % this template was split off from another, so rejoin it
        rez.st3(current_cluster == ik, 7) = split_orig_template(ik);
    end
end

nCluOld = numel(unique(current_cluster));
nCluNew = numel(unique(split_orig_template));
fprintf('Reduced number of clusters from %d to %d\n', nCluOld, nCluNew);

end