import cortex
import numpy as np
import nibabel as nib
import time

class InstaCortexViewer(object):
    def __init__(self, subject, xfm_name, html_template='simple.html', port=4567,
                 mask_type='thick', cmap='fingfind_heatmap_2', vmin=-1., vmax=1., **kwargs):
        super(InstaCortexViewer, self).__init__()
        npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()

        self.bufferlen = 2
        data = np.zeros((self.bufferlen, npts), 'float32')
        vol = cortex.Volume(data, subject, xfm_name, cmap=cmap, vmin=vmin, vmax=vmax)
        view = cortex.webshow(vol, port=port, autoclose=False, template=html_template, title='instabrain')

        self.subject = subject
        self.xfm_name = xfm_name
        self.mask_type = mask_type
        # self.functional_dir = '/Users/efun/Dropbox/pycortex-db/subjects/'+self.subject+'/functionals'
        self.functional_dir = '/Users/eo5629/fmri/'+self.subject+'/ref'
        # self.functional_dir = '/Users/efun/Dropbox/pycortex-db/subjects/'+self.subject+'/functionals'

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
        img_dir = self.functional_dir+'/'+img_name+'.nii'
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
    subj_id = 'ff001'
    xfm_name = 'rai2rfi'
    viewer = InstaCortexViewer(subj_id, xfm_name)
    print 'PyCortex started!'
    while True: pass
