classdef InstaLocalizer < handle
    properties
        CONFIG
        base_dir
        ref_dir
        bold_dir
        tr
        vols
        num_runs
        labels = []
        features = []
        trs_trial_end_offset
        trs_per_trial
        moving_average_trs
    end

    methods
        function self = InstaLocalizer(subjid)
            addpath(genpath(pwd));
            self.CONFIG = YAML.read('localizer_config.yml');
            self.base_dir = self.CONFIG.SUBJECT_DIR;
            self.base_dir = [self.base_dir '/' subjid];
            self.ref_dir = strcat(self.base_dir,'/ref');
            self.bold_dir = strcat(self.base_dir,'/bold');
            self.tr = self.CONFIG.TR;
            self.vols = self.CONFIG.VOLS_PER_RUN;
            self.num_runs = self.CONFIG.NUM_RUNS;
            self.trs_trial_end_offset = self.CONFIG.TRS_TRIAL_END_OFFSET;
            self.trs_per_trial = self.CONFIG.TRS_PER_TRIAL;
            self.moving_average_trs = self.CONFIG.MOVING_AVERAGE_TRS;
        end

        function allPreprocessingSteps
            % subject-id = s001
            % download and extract s001 data
            % dcm2niix s001
            % mv *mprage* s001/ref/rai.nii
            % mv *run-00X*.nii s001/bold/run-00X.nii
            % mv [fieldmap] s001/ref/[fieldmap].nii
            % recon-all -s s002fs -i $SUBJECTS_DIR/s002/ref/rai.nii -all
            % P = spm_select('ExtList', pwd, '.*nii');
            % spm_realign(P);
            % spm_reslice(P);
            % mv meanrun_001.nii s001/ref/rfi.nii
            % rrun_XXX.nii 
            % bbregister --s s001fs --mov rfi.nii --init-fsl --bold --reg rai2rfi.dat
            % bbregister --s s002fs --mov rfi.nii --init-fsl --bold --reg rai2rfi.dat
            % mri_label2vol --subject s001fs --label $SUBJECTS_DIR/s001fs/label/lh.BA6_exvivo.label --temp rfi.nii --reg rai2rfi.dat --proj frac 0 1 .1 --fillthresh .3 --hemi lh --o masklhBA6_rfi.nii.gz
        end

        function extractSeqFeatures(self)
            sequences_per_trial = 3;
            time_per_block = self.tr*self.trs_per_trial;
            fid = fopen(strcat(self.ref_dir,'/trial-data.txt'), 'rt');
            raw_trial_data = textscan(fid, '%d %d %f %s %s', 'Delimiter',',','HeaderLines', 1);
            fclose(fid);
            for column = 1:length(raw_trial_data)
                raw_trial_data{column} = raw_trial_data{column}(1:sequences_per_trial:length(raw_trial_data{column}));
            end
            run_nums = cell2mat(raw_trial_data(1));
            trial_nums = cell2mat(raw_trial_data(2));
            raw_labels = raw_trial_data(5);
            raw_labels = raw_labels{:}; % for quicker access ;)
            unique_labels = unique(raw_labels);
            self.labels = zeros(length(raw_labels),1); % set labels based on trial data
            for label = 1:length(unique_labels)
                self.labels(find(ismember(raw_labels,unique_labels(label)))) = label;
            end

            self.features = [];
            for run_num = 1:self.num_runs
                raw_img = spm_read_vols(spm_vol([self.bold_dir sprintf('/rrun-%3.3d.nii',run_num)]));
                raw_img_flat = reshape(raw_img,[],size(raw_img,4));
                dt_img_flat = detrend(raw_img_flat')';
                zs_dt_img_flat = zscore(dt_img_flat')';
                avg_filter_a = 1;
                avg_filter_b = ones(1,self.moving_average_trs)/self.moving_average_trs;
                % this does moving average of TR and n-1 previous TRs
                filt_zs_dt_img_flat = filter(avg_filter_b, avg_filter_a, zs_dt_img_flat')';

                for data_sample = trial_nums(find(run_nums==run_num))'
                    % TR starts at last TR of trial, minus trial_end_offset
                    % (which has been averaged backwards in time by n-1 TRs)
                    tr = self.trs_trial_end_offset + data_sample*self.trs_per_trial;
                    self.features = [self.features filt_zs_dt_img_flat(:,tr)];
                end
            end
        end

        function extractFingerFeatures(self)
            presses_per_trial = 10;
            time_per_block = self.tr*self.trs_per_trial;
            fid = fopen(strcat(self.ref_dir,'/trial-data.txt'), 'rt');
            raw_trial_data = textscan(fid, '%f %d %d %f %d %d %d %d', 'Delimiter',' ','HeaderLines', 1);
            fclose(fid);
            for column = 1:length(raw_trial_data)
                raw_trial_data{column} = raw_trial_data{column}(1:presses_per_trial:length(raw_trial_data{column}));
            end
            run_nums = cell2mat(raw_trial_data(5));
            run_nums = run_nums+1;
            trial_nums = cell2mat(raw_trial_data(8));
            trial_nums = trial_nums/10+1;
            raw_labels = raw_trial_data(2);
            raw_labels = raw_labels{:};
            raw_labels = raw_labels((length(raw_labels)/2+1):end);
            unique_labels = unique(raw_labels);
            self.labels = zeros(length(raw_labels),1); % set labels based on trial data
            for label = 1:length(unique_labels)
                self.labels(find(ismember(raw_labels,unique_labels(label)))) = label;
            end

            self.features = [];
            % for run_num = 1:self.num_runs
            for run_num = (self.num_runs/2+1):self.num_runs
                raw_img = spm_read_vols(spm_vol([self.bold_dir sprintf('/rrun-%3.3d.nii',run_num)]));
                raw_img_flat = reshape(raw_img,[],size(raw_img,4));
                dt_img_flat = detrend(raw_img_flat')';
                zs_dt_img_flat = zscore(dt_img_flat')';
                avg_filter_a = 1;
                avg_filter_b = ones(1,self.moving_average_trs)/self.moving_average_trs;
                % this does moving average of TR and n-1 previous TRs
                filt_zs_dt_img_flat = filter(avg_filter_b, avg_filter_a, zs_dt_img_flat')';

                for data_sample = trial_nums(find(run_nums==run_num))'
                    % TR starts at last TR of trial, minus trial_end_offset
                    % (which has been averaged backwards in time by n-1 TRs)
                    tr = self.trs_trial_end_offset + data_sample*self.trs_per_trial;
                    self.features = [self.features filt_zs_dt_img_flat(:,tr)];
                end
            end
        end

        function extractSrtSeries(self)
            for run_num = 1:self.num_runs
                raw_img = spm_read_vols(spm_vol([self.bold_dir sprintf('/hrrun-%3.3d.nii',run_num)]));
                raw_img_flat = reshape(raw_img,[],size(raw_img,4));
                dt_img_flat = detrend(raw_img_flat')';
                zs_dt_img_flat = zscore(dt_img_flat')';
                % self.features = [self.features zs_dt_img_flat];

                avg_filter_a = 1;
                avg_filter_b = ones(1,self.moving_average_trs)/self.moving_average_trs;
                % this does moving average of TR and n-1 previous TRs
                filt_zs_dt_img_flat = filter(avg_filter_b, avg_filter_a, zs_dt_img_flat')';
                self.features = [self.features filt_zs_dt_img_flat];

                % for data_sample = trial_nums(find(run_nums==run_num))'
                %     % TR starts at last TR of trial, minus trial_end_offset
                %     % (which has been averaged backwards in time by n-1 TRs)
                %     tr = self.trs_trial_end_offset + data_sample*self.trs_per_trial;
                %     self.features = [self.features filt_zs_dt_img_flat(:,tr)];
                % end
            end
        end

        function extractFtSeries(self)
            % for run_num = 1:self.num_runs
            for run_num = 1:(self.num_runs/2)
                raw_img = spm_read_vols(spm_vol([self.bold_dir sprintf('/rrun-%3.3d.nii',run_num)]));
                raw_img_flat = reshape(raw_img,[],size(raw_img,4));
                dt_img_flat = detrend(raw_img_flat')';
                zs_dt_img_flat = zscore(dt_img_flat')';
                % self.features = [self.features zs_dt_img_flat];

                avg_filter_a = 1;
                avg_filter_b = ones(1,self.moving_average_trs)/self.moving_average_trs;
                % this does moving average of TR and n-1 previous TRs
                filt_zs_dt_img_flat = filter(avg_filter_b, avg_filter_a, zs_dt_img_flat')';
                self.features = [self.features filt_zs_dt_img_flat];
            end
        end

    end

    methods(Static)
        function out_data = processingStep(in_data)
            % code
        end
    end
end
