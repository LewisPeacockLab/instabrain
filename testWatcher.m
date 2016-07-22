classdef testWatcher < handle
    properties
    watch_dir = '/Users/efun/seqlearn/bold/split/'
    file_ext = '*.nii'
    % file_ext = '.imgdat'
    read_files = {}
    end

    methods
        function self = testWatcher
            % initialization here
        end

        function run(self)
            while true
                % code
            end
        end

        function check_dir(self)
            file_list = dir(strcat(self.watch_dir,self.file_ext));
            file_names = {};
            for index = 1:size(file_list,1)
                file_names{size(file_names,2)+1} = file_list(index).name;
            end
            self.read_files = file_names;
        end

        function check_dir_pseudocode(self)
            % 1. get list of .imgdat files in watch_dir
            % 2. compare to in_process files in watch_dir
            % 3. for each file, if not in in_process list:
            %    - add to in_process list
            %    - start processing
            % 4. after processing:
            %    - move to watch_dir/processed
            %    - remove from in_process list
            %    - increment dict/counter
        end
    end
end
