# realtime
from insta_ff_localizer import InstaLocalizer; sid1 = 'ft001'; sid2 = 'ft002';
loc1 = InstaLocalizer(sid1); loc2 = InstaLocalizer(sid2);
roi_name='ba4a'; loc1.extract_features(roi_name); loc2.extract_features(roi_name)
roi_name='ba4a'; loc1.extract_features(roi_name,zs_all=False,detrend=True); loc2.extract_features(roi_name,zs_all=False,detrend=True)
roi_name='ba4a'; loc1.extract_features(roi_name,zs_all=False,detrend=False); loc2.extract_features(roi_name,zs_all=False,detrend=False)

# load classifier
import pickle
clf1 = pickle.load(open(loc1.ref_dir+'/clf.p'))
clf2 = pickle.load(open(loc2.ref_dir+'/clf.p'))

np.mean(clf1.predict(loc1.fmri_data)==loc1.fmri_data.targets)
np.mean(clf2.predict(loc2.fmri_data)==loc2.fmri_data.targets)

# localizer
from insta_localizer import InstaLocalizer; sid1 = 'ft001'; sid2 = 'ft002';
loc1 = InstaLocalizer(sid1); loc2 = InstaLocalizer(sid2);
roi_name='ba4'; hemi='rh'; loc1.extract_features(roi_name,hemi); loc2.extract_features(roi_name,hemi)
loc1.cross_validate(5); loc2.cross_validate(5)
print np.mean(loc1.out_accs);print np.std(loc1.out_accs)/2.;print np.mean(loc2.out_accs);print np.std(loc2.out_accs)/2.;

