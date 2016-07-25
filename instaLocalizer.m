% stage 1:
% a.) freesurfer recon-all
% b.) spm preprocessing
%     - alignment (anatomical<->func with flirt -bbr)
%     - motion correction (func<->func with mcflirt)

% stage 2: 
% a.) freesurfer ROI selection
% b.) transform ROIs to RFI space

% stage 3:
% model generation
% SPM t-stats map

% stage 4:
% classifier generation
