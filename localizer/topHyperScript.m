s1 = 'hs001'; s2 = 'hs002'; mask_id = 1;
loc1 = InstaLocalizer(s1); loc2 = InstaLocalizer(s2);
loc11 = InstaLocalizer(s1); loc22 = InstaLocalizer(s2);
loc111 = InstaLocalizer(s1); loc222 = InstaLocalizer(s2);
disp('extracting SRT time series...')
loc111.extractSrtSeries; loc222.extractSrtSeries;
disp('extracting FT time series...')
loc1.extractFtSeries; loc2.extractFtSeries;
disp('extracting FT features...')
loc11.extractFingerFeatures; loc22.extractFingerFeatures;
disp('training classifier...')
clf1 = InstaFtClassifier(s1); clf2 = InstaFtClassifier(s2);
% clf1.loadSeriesData([loc1.features loc111.features],clf1.mask_names(mask_id));
% clf2.loadSeriesData([loc2.features loc222.features],clf2.mask_names(mask_id));
clf1.loadSeriesData(loc1.features,clf1.mask_names(mask_id));
clf2.loadSeriesData(loc2.features,clf2.mask_names(mask_id));
clf1.hyperAlign(clf2);
clf1.trainHyperClassifier(loc11.features,loc22.features,clf2,loc11.labels,loc22.labels);
