classdef InstaClassifier < handle
    properties
    % properties here
    N_training_samples = 100 % samples per class
    N_test_samples = 100 % samples per class
    N_class = 8
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

        function sampleClassifier(self, brain)
            self.fmri_data = [];
            self.labels = [];
            for c = 1:self.N_class
                fmri_one_class = [];
                orientation = 22.5*(c-1);
                for vol = 1:self.N_training_samples
                    fmri_one_class = [fmri_one_class; brain.sampleNoisyVolume(orientation)'];
                end
                self.fmri_data = [self.fmri_data; fmri_one_class];
                self.labels = [self.labels; c*ones(self.N_training_samples,1)];
            end

            [ixtr,ixte] = separate_train_test(self.labels, 0.5);

            [self.weights, self.ix_eff, self.errTable_tr, self.errTable_te] = muclsfy_smlr(...
                self.fmri_data(ixtr,:), self.labels(ixtr,:), self.fmri_data(ixte,:), self.labels(ixte,:),...
                'wdisp_mode', 'iter', 'nlearn', 300, 'mean_mode', 'none', 'scale_mode', 'none');
        end

        function mean_class_probs = testClassifierSim(self, brain)
            % update to include variance of prob per orientation
            % sometimes gets NaN's - investigate
            mean_class_probs = zeros(self.N_class,self.N_test_samples);
            for orientation = 1:self.N_class
                for sample = 1:self.N_test_samples
                    all_orientation_probs = self.applyClassifier(brain.sampleVolume(22.5*(orientation-1)));
                    mean_class_probs(orientation,sample) = all_orientation_probs(orientation);
                end
            end
            mean_class_probs = mean(mean_class_probs,2);
        end

        function [problem_brain, problem_probs] = testClassifierNan(self, brain)
            for orientation = 1:self.N_class
                for sample = 1:self.N_test_samples
                    new_brain = brain.sampleNoisyVolume(22.5*(orientation-1));
                    all_orientation_probs = self.applyClassifier(new_brain);
                    mean_class_probs(orientation,sample) = all_orientation_probs(orientation);
                    if isnan(mean_class_probs(orientation,sample))
                       disp('problem found'); 
                        problem_brain = new_brain;
                        problem_probs = all_orientation_probs;
                        return
                    end
                end
            end
            disp('no problem');
            problem_brain = 0;
            problem_probs = 0;
        end

        function [means, vars, maxes, mins] = testClassifierRaw(self, brain)
            class_outs = zeros(self.N_class,self.N_class,self.N_test_samples);
            for orientation = 1:self.N_class
                for sample = 1:self.N_test_samples
                    class_outs(orientation,:,sample) = self.applyClassifierRaw(brain.sampleNoisyVolume(22.5*(orientation-1)));
                end
            end
            means = mean(class_outs,3);
            vars = var(class_outs,0,3);
            maxes = max(class_outs,[],3);
            mins = min(class_outs,[],3);
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
