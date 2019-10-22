% mexGPUall. For these to complete succesfully, you need to configure the
% Matlab GPU library first (see README files for platform-specific
% information)

    mexcuda -largeArrayDims mexThSpkPC.cu
    mexcuda -largeArrayDims mexGetSpikes2.cu
    mexcuda -largeArrayDims mexMPnu8.cu

    mexcuda -largeArrayDims mexSVDsmall2.cu
    mexcuda -largeArrayDims mexWtW2.cu
    mexcuda -largeArrayDims mexFilterPCs.cu
    mexcuda -largeArrayDims mexClustering2.cu
    mexcuda -largeArrayDims mexDistances2.cu

    % @djoshea reproducible versions of some of the above, used when ops.reproducible is true
    mexcuda -largeArrayDims mexThSpkPCr.cu
    mexcuda -largeArrayDims mexGetSpikes2r.cu
    mexcuda -largeArrayDims mexMPnu8r.cu
    mexcuda -largeArrayDims mexSVDsmall2r.cu
    mexcuda -largeArrayDims mexClustering2r.cu
    
%    mex -largeArrayDims mexMPmuFEAT.cu
%    mex -largeArrayDims mexMPregMU.cu
%    mex -largeArrayDims mexWtW2.cu

% If you get uninterpretable errors, run again with verbose option -v, i.e. mexcuda -v largeArrayDims mexGetSpikes2.cu


