classdef InstaLocalizer < handle
    properties
        CONFIG
        base_dir
        ref_dir
        bold_dir
        spm_dir
        spm_jobs
        freesurfer_subj_dir
        tr
        vols
        num_runs
    end

    methods
        function self = instaLocalizer
            addpath(genpath(pwd));
            self.CONFIG = YAML.read('localizer_config.yml');
            self.base_dir = self.CONFIG.SUBJECT_DIR;
            self.ref_dir = strcat(self.base_dir,'/ref');
            self.bold_dir = strcat(self.base_dir,'/bold');
            self.spm_dir = strcat(self.ref_dir,'/spm');
            self.freesurfer_subj_dir = strcat(self.ref_dir,'/freesurfer');
            setenv('FREESURFER_HOME', self.CONFIG.FREESURFER_HOME);
            setenv('FSLDIR', self.CONFIG.FSL_DIR);
            setenv('FSL_DIR', self.CONFIG.FSL_DIR);
            setenv('SUBJECTS_DIR', self.freesurfer_subj_dir);

            self.tr = self.CONFIG.TR;
            self.vols = self.CONFIG.VOLS_PER_RUN;
            self.num_runs = self.CONFIG.NUM_RUNS;
            % http://andysbrainblog.blogspot.com/2013/10/whats-in-spmmat-file.html 
        end

        function allPreprocessingSteps
            % fs recon-all
            % bbregister
            % create RFI by motion correcting middle run
            % motion correct all other runs to RFI
            % P = spm_select('ExtList', pwd, '^ar01.nii', 1:165);
            % spm_realign(P);
            % spm_reslice(P);
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
            out_trial_data{1} = raw_trial_data{1};
            out_trial_data{2} = raw_trial_data{5};
            out_trial_data{3} = time_per_block*[raw_trial_data{2}-1];
            out_trial_data{4} = time_per_block*ones(length(out_trial_data{1}),1);
        end

        function setUpGlm(self)
            funcs = {};
            for runIdx = 1:self.num_runs
               % funcs(length(funcs)+1) = cellstr([self.bold_dir '/run_' sprintf('%03d', 5) '_mc.nii']);
               funcs(length(funcs)+1) = cellstr(['run_' sprintf('%03d', 5) '_mc.nii']);
            end

            self.spm_jobs{1}.stats{1}.fmri_spec.dir = cellstr(self.spm_dir);
            self.spm_jobs{1}.stats{1}.fmri_spec.timing.units = 'secs'; % 'scans' or 'secs'
            self.spm_jobs{1}.stats{1}.fmri_spec.timing.RT = self.tr;
            self.spm_jobs{1}.stats{1}.fmri_spec.timing.fmri_t = 16;
            self.spm_jobs{1}.stats{1}.fmri_spec.timing.fmri_t0 = 0;
            self.spm_jobs{1}.stats{1}.fmri_spec.fact = struct('name', {}, 'levels', {});
            self.spm_jobs{1}.stats{1}.fmri_spec.bases.hrf = struct('derivs', [0 0]);
            self.spm_jobs{1}.stats{1}.fmri_spec.volt = 1;
            self.spm_jobs{1}.stats{1}.fmri_spec.global = 'None';
            self.spm_jobs{1}.stats{1}.fmri_spec.mask = {''};
            self.spm_jobs{1}.stats{1}.fmri_spec.cvi = 'AR(1)';

            % T = textscan(fid, '%f %s %f %f', 'HeaderLines', 1); %Columns should be 1)Run, 2)Regressor Name, 3) Onset Time (in seconds, relative to start of each run), and 4)Duration, in seconds
            T = self.loadSequenceRegs;

            for runIdx = 1:self.num_runs
                    nameList = unique(T{2});
                    names = nameList';
                    onsets = cell(1, size(nameList,1));
                    durations = cell(1, size(nameList,1));
                    sizeOnsets = size(T{3}, 1);
                for nameIdx = 1:size(nameList,1)
                    for idx = 1:sizeOnsets
                        if isequal(T{2}{idx}, nameList{nameIdx}) && T{1}(idx) == runIdx
                            onsets{nameIdx} = double([onsets{nameIdx} T{3}(idx)]);
                            durations{nameIdx} = double([durations{nameIdx} T{4}(idx)]);
                        end
                    end
                end

                save([self.spm_dir '/reg_' num2str(runIdx)], 'names', 'onsets', 'durations')
                
                files = spm_select('ExtFPList', [self.bold_dir], ['^' funcs{runIdx}], 1:self.vols);

                self.spm_jobs{1}.stats{1}.fmri_spec.sess(runIdx).scans = cellstr(files);
                self.spm_jobs{1}.stats{1}.fmri_spec.sess(runIdx).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
                self.spm_jobs{1}.stats{1}.fmri_spec.sess(runIdx).multi = cellstr([self.spm_dir '/reg_' num2str(runIdx) '.mat']);
                self.spm_jobs{1}.stats{1}.fmri_spec.sess(runIdx).regress = struct('name', {}, 'val', {});
                self.spm_jobs{1}.stats{1}.fmri_spec.sess(runIdx).multi_reg = {''};
                self.spm_jobs{1}.stats{1}.fmri_spec.sess(runIdx).hpf = 128;
            end
        end

        function specifyGlm(self)
            cd(self.spm_dir);
            spm_jobman('run', self.spm_jobs);
        end

        function estimateGlm(self)
            cd(self.spm_dir);
            load SPM;
            spm_spm(SPM);
        end

        function allStatModelSteps
            % load model
            % glm = spm12w_getp('type','glm', 'sid',args.sid, 'para_file',args.glm_file);
            % glm = spm12w_glm_build('type',mfield{1},'params',glm); 
            % glm.SPM = spm12w_getspmstruct('type','glm','params',glm);
            % glm.SPM = spm_fmri_spm_ui(glm.SPM);
            % glm.SPM = spm_spm(glm.SPM);
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
% mri_tkregister2 --mov "template path" --s "subject id" --regheader --reg ./register.dat *that's the output file*
% mri_label2vol --subject "subject id" --label "subject path"/label/"lh|rh"."BA6|BA4a|BA4p".label --temp "template path" --reg register.dat --proj frac 0 1 .1 --fillthresh .3 --hemi "lh|rh" --o "mask".nii.gz
% mri_label2vol --subject seqlearn-003 --label $SUBJECTS_DIR/seqlearn-003/label/lh.BA4a.label --temp rfi.nii.gz --reg test_new_reg.dat --proj frac 0 1 .1 --fillthresh .3 --hemi lh --o masklhBA4a_bbr_rfi.nii.gz
