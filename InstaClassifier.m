classdef InstaClassifier < handle
    properties
    N_class = 4
    fmri_data
    labels
    weights
    ix_eff
    errTable_tr
    errTable_te
    end

    methods
        function self = SimfeedClassifier
            % initialization here
        end

        function trainClassifier(self)
            basedir = '/belly/20160615-seqlearn-003/ref';
            betas_img = spm_read_vols(spm_vol([basedir '/betas.nii']));
            mask_img = spm_read_vols(spm_vol([basedir '/masklhBA4a_bbr_rfi.nii']));
            betas_img_flat = reshape(betas_img,[],size(betas_img,4));
            mask_img_flat = reshape(mask_img,[],1);
            roi_map = zeros(length(mask_img_flat),sum(mask_img_flat>0));
            roi_voxel = 1;
            for brain_voxel = 1:length(mask_img_flat)
                if mask_img_flat(brain_voxel)>0
                    roi_map(brain_voxel,roi_voxel) = 1;
                    roi_voxel = roi_voxel + 1;
                end
            end
            self.fmri_data = betas_img_flat'*roi_map;
            self.labels = repmat([1 2 3 4],[1 8])';
            [ixtr,ixte] = separate_train_test(self.labels, 0.5);

            [self.weights, self.ix_eff, self.errTable_tr, self.errTable_te] = muclsfy_smlr(...
                self.fmri_data(ixtr,:), self.labels(ixtr,:), self.fmri_data(ixte,:), self.labels(ixte,:),...
                'wdisp_mode', 'iter', 'nlearn', 300, 'mean_mode', 'none', 'scale_mode', 'none');
        end

        function class_probs = applyClassifier(self, fmri_data)
            % [tmp, label_est] = max(fmri_data' * self.weights(1:1000,:),[],2);
            eY = exp(fmri_data'*self.weights(1:1000,:)); % Nsamp*Nclass
            class_probs = eY ./ repmat(sum(eY,2), [1, self.N_class]); % Nsamp*Nclass
        end

        function class_out = applyClassifierRaw(self, fmri_data)
            class_out = fmri_data'*self.weights(1:1000,:);
        end
    end

    methods(Static)
        function score = tuneToScore(class_probs, tuning_curve, target_class)
            score = class_probs*circshift(tuning_curve,[0 target_class-1])';
        end
    end
end
