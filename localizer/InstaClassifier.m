classdef InstaClassifier < handle
    properties
    N_class = 3
    base_dir = '/Users/eo5629/fmri/seqlearn/s001'
    fmri_data
    mask_img
    mask_dims
    mask_map
    mask_header
    labels
    weights
    ix_eff
    errTable_tr
    errTable_te
    mask_names = {'maskrhBA4a_rfi', 'maskrhBA4p_rfi', 'maskrhBA6_rfi', 'masklhBA4a_rfi', 'masklhBA4p_rfi', 'masklhBA6_rfi', 'maskbiBA4a_rfi', 'maskbiBA4p_rfi', 'maskbiBA6_rfi'}
    cross_validation
    selection_count
    end

    methods
        function self = InstaClassifier
            % one ROI at a time for classifier
        end

        function loadMask(self, name)
            self.mask_header = spm_vol([self.base_dir '/ref/' name '.nii']);
            self.mask_img = spm_read_vols(self.mask_header);
            self.mask_dims = size(self.mask_img);
            self.mask_img = reshape(self.mask_img,[],1);
        end

        function saveClassifier(self, name)
            out_header = self.mask_header;
            out_header.dt = [16 0];
            out_img_all_classes = self.mask_map*self.weights(1:(size(self.weights,1)-1),:);
            for class_type = 1:self.N_class
                out_header.fname = [self.base_dir '/ref/' name '-class-' num2str(class_type) '.nii'];
                class_img = out_img_all_classes(:,class_type);
                class_img = reshape(class_img, self.mask_dims);
                spm_write_vol(out_header, class_img);
            end
            % fslmerge -t clf.nii clf*; gunzip *.gz
        end

        function trainClassifier(self, features, labels, mask, train_ratio)
            % clf.trainClassifier(loc.features,loc.labels,clf.mask_names(4),.5)
            self.loadMask(char(mask));
            self.mask_map = zeros(length(self.mask_img),sum(self.mask_img>0));
            roi_voxel = 1;
            for brain_voxel = 1:length(self.mask_img)
                if self.mask_img(brain_voxel)>0
                    self.mask_map(brain_voxel,roi_voxel) = 1;
                    roi_voxel = roi_voxel + 1;
                end
            end
            self.fmri_data = features'*self.mask_map;
            self.labels = labels;
            [ixtr,ixte] = separate_train_test(self.labels, train_ratio);

            [self.weights, self.ix_eff, self.errTable_tr, self.errTable_te] = muclsfy_smlr(...
                self.fmri_data(ixtr,:), self.labels(ixtr,:), self.fmri_data(ixte,:), self.labels(ixte,:),...
                'wdisp_mode', 'iter', 'nlearn', 30, 'mean_mode', 'none', 'scale_mode', 'none');
        end

        function calcSelectionCount(self, features, labels, mask, iters, train_ratio)
            self.loadMask(char(mask));
            self.mask_map = zeros(length(self.mask_img),sum(self.mask_img>0));
            roi_voxel = 1;
            for brain_voxel = 1:length(self.mask_img)
                if self.mask_img(brain_voxel)>0
                    self.mask_map(brain_voxel,roi_voxel) = 1;
                    roi_voxel = roi_voxel + 1;
                end
            end
            self.fmri_data = features'*self.mask_map;
            self.labels = labels;
            for nn = 1:iters
                fprintf('\n\nCross Validation Trial : %3d \n', nn)
                [ixtr, ixte] = separate_train_test(self.labels, train_ratio);
                [ww, ix_eff, errTable_tr, errTable_te, parms] = muclsfy_smlr(...
                    self.fmri_data(ixtr,:), self.labels(ixtr,:), self.fmri_data(ixte,:), self.labels(ixte,:),...
                    'wdisp_mode', 'iter', 'nlearn', 300, 'mean_mode', 'none', 'scale_mode', 'none');
                CVRes(nn).ix_eff_all = ix_eff;
                CVRes(nn).errTable_te = errTable_te;
                CVRes(nn).errTable_tr = errTable_tr;
                CVRes(nn).g = parms;
            end

            % N-value 
            self.cross_validation = CVRes;
            self.selection_count = calc_SCval(CVRes, {'Survived'});
        end

        function class_probs = applyClassifier(self, fmri_data)
            eY = exp(fmri_data'*self.weights(1:(size(self.weights,1)-1),:)); % Nsamp*Nclass
            class_probs = eY ./ repmat(sum(eY,2), [1, self.N_class]); % Nsamp*Nclass
        end

        function class_out = applyClassifierRaw(self, fmri_data)
            class_out = fmri_data'*self.weights(1:(size(self.weights,1)-1),:);
        end
    end

    methods(Static)
        function score = tuneToScore(class_probs, tuning_curve, target_class)
            score = class_probs*circshift(tuning_curve,[0 target_class-1])';
        end
    end
end
