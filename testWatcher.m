classdef testWatcher < handle
    properties
        watch_dir = '~/code/instabrain/data/'
        file_ext = '*.imgdat'
        match_dir
        read_files = {}
        in_process_files = {}
        % pool = gcp()
        in_process_slices = []
        in_process_volumes = []
        fmri_data = [] % (rep,slice,value)
        slice_dim = 96 % or 64, usually
    end

    methods
        function self = testWatcher
            self.match_dir = strcat(self.watch_dir,self.file_ext);
        end

        function run(self)
            while true
                % code
            end
        end

        function checkDir(self)
            all_files = dir(self.match_dir);
            all_files = {all_files.name};
            to_process_list = all_files(not(ismember(all_files,self.in_process_files)));
            for slice = to_process_list
                slice = char(slice);
                rep_num = str2num(cell2mat(regexp(slice,'(?<=-R).*(?=-E)','match')));
                slice_num = str2num(cell2mat(regexp(slice,'(?<=-S).*(?=.imgdat)','match')));
                self.fmri_data(rep_num,slice_num,:) = self.readSlice(strcat(self.watch_dir,slice));
                %    - add to in_process list
                %    - start processing
            end
        end

        function showSlice(self, rep, slice)
            vis_slice = reshape(self.fmri_data(rep,slice,:),self.slice_dim,self.slice_dim);
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
        % 4. after processing:
        %    - move to watch_dir/processed
        %    - remove from in_process list
        %    - increment dict/counter
    end
end
