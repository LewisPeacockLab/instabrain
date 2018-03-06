import cortex
import numpy as np
import nibabel as nib
import time
subj_id = 'fp001'
functional_dir = '/Users/eo5629/pycortex/filestore/db/'+subj_id+'/functionals'
# functional_dir = '/Users/efun/pycortex/filestore/db/'+subj_id+'/functionals'

rfi_img = cortex.Volume(functional_dir+'/rfi.nii', subject=subj_id, xfmname='rai2rfi')
volume = rfi_img

# cortex.webshow(data=volume, template='insta_diagnostics.html') #, recache=True)

# vol_ex_1 = cortex.Volume(functional_dir+'/vol1.nii', subject=subj_id, xfmname='rai2rfi')
# vol_ex_2 = cortex.Volume(functional_dir+'/vol2.nii', subject=subj_id, xfmname='rai2rfi')
# # for multi volumes, currently have to click to refresh UI
# volumes = {
#     'Reference Functional': rfi_img,
#     'Vol 1': vol_ex_1,
#     'Vol 2': vol_ex_2}
# cortex.webshow(data=volumes, template='insta_diagnostics.html', title='instabrain')

# a port number will then be output, for example "Started server on port 39140"
# the viewer can then be accessed in a web browser, in this case at "localhost:39140"

# Demo class for realtime updates

class InstaCortexViewer(object):
    # bufferlen = 50
    bufferlen = 2

    def __init__(self, subject, xfm_name, html_template='simple.html', port=4567,
                 mask_type='thick', vmin=-1., vmax=1., **kwargs):
        super(InstaCortexViewer, self).__init__()
        npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()

        data = np.zeros((self.bufferlen, npts), 'float32')
        vol = cortex.Volume(data, subject, xfm_name, vmin=vmin, vmax=vmax)
        view = cortex.webshow(vol, port=port, autoclose=False, template=html_template, title='instabrain')

        self.subject = subject
        self.xfm_name = xfm_name
        self.mask_type = mask_type

        self.view = view
        self.active = True
        self.i = 0
        self.update_volume(init=True)

    def update_volume(self, img_name='rfi', init=False):
        mos, data = self.load_nib_to_mos(img_name)
        self.view.dataviews.data.data[0]._setData(0, mos)
        self.view.dataviews.data.data[0]._setData(1, mos)
        if init:
            self.view.setVminmax(np.percentile(data,75),np.percentile(data,99))
        self.view.setFrame(1)

    def load_nib_to_mos(self, img_name):
        img_dir = functional_dir+'/'+img_name+'.nii'
        # data = np.array(nib.load(img_dir).get_data().swapaxes(0,2),'float32')+7500*np.random.random((36,100,100))
        data = np.array(nib.load(img_dir).get_data().swapaxes(0,2),'float32')
        vol = cortex.Volume(data, self.subject, self.xfm_name)
        mos, _ = cortex.mosaic(vol.volume[0], show=False)
        return mos, data

    def generate_random_mos(self):
        data = 15000+10000*np.random.random((36,100,100))
        vol = cortex.Volume(data, self.subject, self.xfm_name)
        mos, _ = cortex.mosaic(vol.volume[0], show=False)
        return mos

    def update_volume_smooth(self, img_name):
        mos, _ = self.load_nib_to_mos(img_name)
        self.view.dataviews.data.data[0]._setData(1, mos)
        self.view.setFrame(0)
        self.view.playpause('play')
        time.sleep(1)
        self.view.playpause('pause')
        self.view.setFrame(1)
        self.view.dataviews.data.data[0]._setData(0, mos)

    def show_run_smooth(self, tr=2, debug_timing=False):
        for img in range(150):
            img_name = 'vol'+str(img).zfill(4)
            start_time = time.time()
            self.update_volume_smooth(img_name)
            if debug_timing:
                print 'volume update: '+str(time.time()-start_time)+'s'
                print 'waiting: '+str(max(0,tr-(time.time()-start_time)))+'s'
            time.sleep(max(0,tr-(time.time()-start_time)))

    def show_run(self, tr=2, debug_timing=False):
        for img in range(150):
            img_name = 'vol'+str(img).zfill(4)
            start_time = time.time()
            self.update_volume(img_name)
            if debug_timing:
                print 'volume update: '+str(time.time()-start_time)+'s'
                print 'waiting: '+str(max(0,tr-(time.time()-start_time)))+'s'
            time.sleep(max(0,tr-(time.time()-start_time)))

    def stop(self):
        self.view.playpause('pause')

if __name__ == '__main__':
    viewer = InstaCortexViewer(subj_id, 'rai2rfi') #, port=4567)
    print 'Instabrain started!'
    while True: pass

###########################################
# converting from freesurfer to pycortex: #
###########################################

# cortex.freesurfer.import_subj('fp001fs', sname='fp001')
# cortex.align.manual('fp001', 'testreg',reference='rfi.nii')
# cortex.align.automatic('fp001', 'testreg',reference='rfi.nii')
# tkregister2 --mov functionals/rfi.nii --targ anatomicals/raw.nii.gz --reg transforms/rai2rfi.dat --fslregout transforms/rai2rfi.mat --noedit
# from cortex.xfm import Transform; import numpy as np
# x = np.loadtxt('transforms/rai2rfi.mat')
#        # Pass transform as FROM epi TO anat; transform will be inverted
#        # back to anat-to-epi, standard direction for pycortex internal
#        # storage by from_fsl
# Transform.from_fsl(x, 'functionals/rfi.nii', 'anatomicals/raw.nii.gz').save('fp001', 'rai2rfi', 'coord')
