classdef InstaLocalizer < handle
    properties
        CONFIG
        base_dir
        ref_dir
        bold_dir
        freesurfer_subj_dir
        tr
        vols
        num_runs
        labels = []
        features = []
        trs_trial_start_offset = 0
        trs_per_trial = 5
    end

    methods
        function self = InstaLocalizer
            addpath(genpath(pwd));
            self.CONFIG = YAML.read('localizer_config.yml');
            self.base_dir = self.CONFIG.SUBJECT_DIR;
            self.ref_dir = strcat(self.base_dir,'/ref');
            self.bold_dir = strcat(self.base_dir,'/bold');
            self.freesurfer_subj_dir = strcat(self.ref_dir,'/freesurfer');
            setenv('FREESURFER_HOME', self.CONFIG.FREESURFER_HOME);
            setenv('FSLDIR', self.CONFIG.FSL_DIR);
            setenv('FSL_DIR', self.CONFIG.FSL_DIR);
            setenv('SUBJECTS_DIR', self.freesurfer_subj_dir);

            self.tr = self.CONFIG.TR;
            self.vols = self.CONFIG.VOLS_PER_RUN;
            self.num_runs = self.CONFIG.NUM_RUNS;
        end

        function allPreprocessingSteps
            % fs recon-all
            % bbregister
            % P = spm_select('ExtList', pwd, '.*nii');
            % spm_realign(P);
            % spm_reslice(P);
            % meanrun_001.nii
            % rrun_XXX.nii 
        end

        function out_trial_data = loadSequenceRegs(self)
            sequences_per_trial = 3;
            time_per_block = self.tr*5;
            fid = fopen(strcat(self.ref_dir,'/trial_data.txt'), 'rt');
            raw_trial_data = textscan(fid, '%d %d %f %s %s', 'Delimiter',',','HeaderLines', 1);
            fclose(fid);
            for column = 1:length(raw_trial_data)
                raw_trial_data{column} = raw_trial_data{column}(1:sequences_per_trial:length(raw_trial_data{column}));
            end
            % out_trial_data{1} = raw_trial_data{1};
            % out_trial_data{2} = raw_trial_data{5};
            % out_trial_data{3} = time_per_block*[raw_trial_data{2}-1];
            % out_trial_data{4} = time_per_block*ones(length(out_trial_data{1}),1);
        end

        function extractFeatures(self)
            sequences_per_trial = 3;
            time_per_block = self.tr*5;
            fid = fopen(strcat(self.ref_dir,'/trial_data.txt'), 'rt');
            raw_trial_data = textscan(fid, '%d %d %f %s %s', 'Delimiter',',','HeaderLines', 1);
            fclose(fid);
            for column = 1:length(raw_trial_data)
                raw_trial_data{column} = raw_trial_data{column}(1:sequences_per_trial:length(raw_trial_data{column}));
            end
            run_nums = cell2mat(raw_trial_data(1));
            trial_nums = cell2mat(raw_trial_data(2));
            raw_labels = raw_trial_data(5);
            raw_labels = raw_labels{:};
            unique_labels = unique(raw_labels);
            self.labels = zeros(length(raw_labels),1); % set labels based on trial data
            for label = 1:length(unique_labels)
                self.labels(find(ismember(raw_labels,unique_labels(label)))) = label;
            end

            self.features = [];
            for run_num = 1:self.num_runs
                raw_img = spm_read_vols(spm_vol([self.bold_dir sprintf('/rrun_%3.3d.nii',run_num)]));
                raw_img_flat = reshape(raw_img,[],size(raw_img,4));
                dt_img_flat = detrend(raw_img_flat')';
                zs_dt_img_flat = zscore(dt_img_flat')';
                avg_filter_a = 1;
                avg_filter_b = [1/3 1/3 1/3];
                filt_zs_dt_img_flat = filter(avg_filter_b, avg_filter_a, zs_dt_img_flat')';

                for data_sample = trial_nums(find(run_nums==run_num))'
                    tr = self.trs_trial_start_offset + data_sample*self.trs_per_trial;
                    self.features = [self.features filt_zs_dt_img_flat(:,tr)];
                end
            end
        end

        function allClassifierSteps
            % train classifier using SPMs?
            % test classifier using shifted volumes, with some thrown out?
        end
    end

    methods(Static)
        function out_data = processingStep(in_data)
            % code
        end
    end
end

% bbregister --s seqlearn-003 --mov rfi.nii.gz --init-fsl --bold --reg test_new_reg.dat
% tkregister2 --mov "template path" --s "subject id" --regheader --reg ./register.dat *that's the output file*
% mri_label2vol --subject "subject id" --label "subject path"/label/"lh|rh"."BA6|BA4a|BA4p".label --temp "template path" --reg register.dat --proj frac 0 1 .1 --fillthresh .3 --hemi "lh|rh" --o "mask".nii.gz
% mri_label2vol --subject seqlearn-003 --label $SUBJECTS_DIR/seqlearn-003/label/lh.BA4a.label --temp rfi.nii.gz --reg test_new_reg.dat --proj frac 0 1 .1 --fillthresh .3 --hemi lh --o masklhBA4a_bbr_rfi.nii.gz
% [temp, est] = max(clf2.applyClassifier(clf1.fmri_data'),[],2);sum( est == loc1.labels)/length(loc1.labels)
% clf2.trainClassifier(loc2.features,loc2.labels,char(clf2.mask_names(1)),.8)
