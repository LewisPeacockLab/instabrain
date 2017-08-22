%%%%%%%%%%%%%%%%%%
% reference data %
%%%%%%%%%%%%%%%%%%
% load reference clf.nii and roi.nii data
% create clf and ROI masks
% load clf to be applied

%%%%%%%%%%%%%%%%%%
% localizer data %
%%%%%%%%%%%%%%%%%%
% load localizer rrun-XXX.nii data
% apply temporal smoothing
% extract training features
% extract mean patterns

%%%%%%%%%%%%%%%%%%%%%%
% neurofeedback data %
%%%%%%%%%%%%%%%%%%%%%%
% load rt-run-XXX.nii data
% apply temporal smoothing
% extract resting state (first XX TRs)
% extract feedback TRs

%%%%%%%%%%%%
% analysis %
%%%%%%%%%%%%
% patterns: mean true, clf weights,
%           resting state (beginning of NFB runs),
%           induction (>0.9 in target clf output), (induction (all)?)
% run correlations between patterns in:
%     - clf voxels only
%     - entire ROI
%
% apply classifier to verify realtime outputs
% - apply to feedback TRs
% - try other motion correction/zscoring and compare results on feedback TRs
% - apply to resting state TRs (defined above)
% - apply to entire time series (for QA)
