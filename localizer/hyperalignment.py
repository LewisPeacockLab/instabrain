from fingfind_localizer import *
from mvpa2.clfs.smlr import SMLR
import numpy as np
all_subjs = np.array([1,2,3,4,5])
target_subj = 1
target_roi = 'ba4a'
NUM_RUNS = 8
clf = SMLR()

non_target_subjs = all_subjs[all_subjs!=target_subj]

locs = {}
for subj in non_target_subjs:
    subj_id = 'ff'+str(subj).zfill(3)
    locs[subj_id] = feature_selection(subj_id, range(NUM_RUNS))

all_runs = np.arange(NUM_RUNS)
for test_run in all_runs:
    target_subj_id = 'ff'+str(target_subj).zfill(3)
    train_runs = all_runs[all_runs!=test_run]
    target_loc = feature_selection(subj_id, train_runs)
    total_fmri_data = []
    all_procrustes = {}
    for subj in non_target_subjs:
        subj_id = 'ff'+str(subj).zfill(3)
        all_procrustes[subj_id] = custom_procrustes(
            target_loc.fmri_data[chunks==train_runs],locs[subj_id].fmri_data[chunks==train_runs])
        total_fmri_data.append(transform_by_procrustes(locs[subj_id],all_procrustes[subj_id]))
    total_fmri_data.append(target_loc.fmri_data[chunks==train_runs])
    clf.train(total_fmri_data)
    clf.predict(target_loc.fmri_data[chunks==test_run])

def custom_procrustes(target_fmri_data, other_fmri_data):
    pass

def transform_by_procrustes(fmri_data, transform):
    pass

def feature_selection(subj_id, chunks):
    localizer = FingfindLocalizer(subj_id)
    localizer.extract_features(target_roi)
    localizer.train_classifier(target_roi, chunks) # need to rewrite this in localizer
    important_voxels = find_important_voxels(localizer.clf.weights)
    important_voxel_locations = localizer.fmri_data.fa.voxel_indices[important_voxels]
    save_mask(subj_id, target_roi, important_voxel_locations)
    localizer.extract_features('hyper_'+target_roi)
    return localizer

def calc_procrustes(X, Y, scaling=True, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

def apply_procrustes(X, tform):
    X_trans = np.matmul(X,tform['rotation'])*tform['scale']+tform['translation']
    return X_trans

def test_procrustes(perturb=False):
    n_features = 2
    n_samples = 10
    angle = np.pi/6.
    translation = np.array([3,-1])
    scaling = 2
    rot_mat = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    X = np.random.random((n_samples,n_features))
    Y = np.matmul(X+translation,rot_mat)*scaling
    if perturb:
        Y += 0.1*np.random.random((n_samples,n_features))
    d,Z,tform = calc_procrustes(X,Y)
    Z_alt = apply_procrustes(Y,tform)
    print 'X:'; print X; print 'Y:'; print Y; print 'Z:'; print Z
    print 'Z_alt:'; print Z_alt; print 'diff:'; print X-Z_alt
