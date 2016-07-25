classdef testWatcher < handle
    properties
        % watch_dir = '~/Dropbox/code/instabrain/data/'
        watch_dir = '~/code/instabrain/data/'
        file_ext = '*.imgdat'
        archive_dir
        match_dir
        read_files = {}
        in_process_files = {}
        pool = gcp()
        in_process_slices = []
        in_process_volumes = []
        fmri_data = [] % (rep,slice,value)
        fmri_data_mc = []
        roi_data_zscore = []
        fmri_data_raw_status = [] % (rep,slice)
        mc_proc_handles = []
        slice_dim = 96 % depends on sequence
        num_slices = 64 % depends on sequence
        volumes_per_run = 50 % depends on protocol
        roi_map = [] % figure out map of fmri voxels to map
        roi_sigmas = []
        roi_voxels = 1000 % figure out count of voxels in ROI
        rfi_data
        last_vol_mc_in_proc
        last_vol_mc_proc
        run_bool = true
        ZSCORE_VOLS = 10
        mc_optimizer
        mc_metric
    end

    methods
        function self = testWatcher
            self.match_dir = strcat(self.watch_dir,self.file_ext);
            self.archive_dir = strcat(self.watch_dir,'archive');
            self.resetForNextRun
            self.rfi_data = zeros(self.num_slices, self.slice_dim, self.slice_dim); % load from ref
            [self.mc_optimizer, self.mc_metric] = imregconfig('monomodal');
        end

        function resetForNextRun(self)
            self.fmri_data = zeros(self.volumes_per_run,self.num_slices,self.slice_dim^2);
            self.fmri_data_mc = zeros(self.volumes_per_run,self.num_slices,self.slice_dim^2);
            self.fmri_data_raw_status = zeros(self.volumes_per_run,self.num_slices);
            self.last_vol_mc_in_proc = 1;
            self.last_vol_mc_proc = 1;
        end

        function update(self)
            self.checkForSlices;
            % self.checkSliceStatus; % add if need to async slice read
            % self.processNewSlices; % add if need to async slice read
            self.checkVolumeStatus;
            self.archiveOldVolumes;
        end

        function checkForSlices(self)
            all_files = dir(self.match_dir);
            all_files = {all_files.name};
            to_process_list = all_files(not(ismember(all_files,self.in_process_files)));
            for slice = to_process_list
                slice = char(slice);
                rep_num = str2num(cell2mat(regexp(slice,'(?<=-R).*(?=-E)','match')));
                slice_num = str2num(cell2mat(regexp(slice,'(?<=-S).*(?=.imgdat)','match')));
                slice_dir = strcat(self.watch_dir,slice);
                self.fmri_data(rep_num,slice_num,:) = self.readSlice(slice_dir);
                self.fmri_data_raw_status(rep_num,slice_num) = 1;
            end
        end

        % function checkSliceStatus(self)
        %     %code
        % end

        % function processNewSlices(self)
        %     %code
        % end

        function checkVolumeStatus(self)
            % do motion correction
            while (all(self.fmri_data_raw_status(self.last_vol_mc_in_proc,:) == 1) && (self.last_vol_mc_in_proc <= self.volumes_per_run))
                tic
                eval_handle = parfeval(self.pool, @imregister, 2,...
                    reshape(self.fmri_data(self.last_vol_mc_in_proc,:,:),self.num_slices,self.slice_dim,self.slice_dim),...
                    self.rfi_data,'rigid', self.mc_optimizer, self.mc_metric);
                toc
                self.mc_proc_handles = [self.mc_proc_handles eval_handle];
                self.last_vol_mc_in_proc = self.last_vol_mc_in_proc + 1;
            end
            % grab motion corrected data
            if length(self.mc_proc_handles>0)
                if strcmp(self.mc_proc_handles(1).State,'finished')
                    mc_vol = fetchOutputs(self.mc_proc_handles(1));
                    self.fmri_data_mc(self.last_vol_mc_proc,:,:) = reshape(mc_vol,self.num_slices,self.slice_dim^2);
                    self.last_vol_mc_proc = self.last_vol_mc_proc + 1;
                    self.mc_proc_handles(1) = [];
                end
            end

            if self.last_vol_mc_proc == self.ZSCORE_VOLS
                self.zScoreCalc;
            end

            if self.last_vol_mc_proc >= self.ZSCORE_VOLS
                self.classifierCalc(self.last_vol_mc_proc);
            end
            % check self.volume_processing_status
        end

        function mc_vol = motionCorrectVolume(self,...
                fmri_data, rfi_data)
            mc_vol = imregister(fmri_data, rfi_data, 'rigid', self.mc_optimizer, self.mc_metric);
        end

        function zScoreCalc(self)
            % self.fmri_data_mc in roi
            % calculate sigmas
        end

        function classifierCalc(self, volume)
            % only within ROI
            % calc linear detrend, apply zscore, and applyClassifier(noise_var, mean, linear_trend)
        end

        function archiveOldVolumes(self)
            %code
        end

        function checkDir(self)
            all_files = dir(self.match_dir);
            all_files = {all_files.name};
            to_process_list = all_files(not(ismember(all_files,self.in_process_files)));
            for slice = to_process_list
                slice = char(slice);
                rep_num = str2num(cell2mat(regexp(slice,'(?<=-R).*(?=-E)','match')));
                slice_num = str2num(cell2mat(regexp(slice,'(?<=-S).*(?=.imgdat)','match')));
                slice_dir = strcat(self.watch_dir,slice);
                self.fmri_data(rep_num,slice_num,:) = self.readSlice(slice_dir);
                % system(['mv ', slice_dir, ' ' self.archive_dir]);
            end
            system(['mv ', self.match_dir, ' ' self.archive_dir]);
        end

        function showSlice(self, rep, slice)
            vis_slice = reshape(self.fmri_data(rep,slice,:),self.slice_dim,self.slice_dim);
            vis_slice = vis_slice/max(max(vis_slice));
            imshow(vis_slice);
        end

        function showSliceMc(self, rep, slice)
            vis_slice = reshape(self.fmri_data_mc(rep,slice,:),self.slice_dim,self.slice_dim);
            vis_slice = vis_slice/max(max(vis_slice));
            imshow(vis_slice);
        end
    end

    methods(Static)
        function slice_data = readSlice(file_name)
            f = fopen(file_name);
            slice_data = fread(f,'uint16');
            fclose(f);
        end

        function roi_voxels = mapVoxelsToRoi(voxels, roi_map)
            roi_voxels = voxels*roi_map;
        end
    end
end
