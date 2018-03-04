import cortex
import numpy as np
subj_id = 'fp001'
# functional_dir = '/Users/eo5629/pycortex/filestore/db/'+subj_id+'/functionals'
functional_dir = '/Users/efun/pycortex/filestore/db/'+subj_id+'/functionals'

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
    bufferlen = 50

    def __init__(self, subject, xfm_name, html_template='simple.html',
                 mask_type='thick', vmin=-1., vmax=1., **kwargs):
        super(InstaCortexViewer, self).__init__()
        npts = cortex.db.get_mask(subject, xfm_name, mask_type).sum()

        data = np.zeros((self.bufferlen, npts), 'float32')
        vol = cortex.Volume(data, subject, xfm_name, vmin=vmin, vmax=vmax)
        view = cortex.webshow(vol, autoclose=False, template=html_template, title='instabrain')
        # view = cortex.webshow(vol, autoclose=False, title='instabrain')

        self.subject = subject
        self.xfm_name = xfm_name
        self.mask_type = mask_type

        self.view = view
        self.active = True
        self.i = 0

    def update_volume(self, mos):
        i, = self.view.setFrame()
        i = round(i)
        new_frame = (i+1) % self.bufferlen
        self.view.dataviews.data.data[0]._setData(new_frame, mos)

    def advance_frame(self):
        i, = self.view.setFrame()
        i = round(i)
        self.view.playpause('play')
        time.sleep(1)
        self.view.playpause('pause')
        self.view.setFrame(i+0.99)

    def run(self, inp):
        if self.active:
            try:
                data = np.fromstring(inp['data'], dtype='float32')
                print(self.subject, self.xfm_name, data.shape)
                vol = cortex.Volume(data, self.subject, self.xfm_name)
                mos, _ = cortex.mosaic(vol.volume[0], show=False)
                self.update_volume(mos)
                self.advance_frame()
    
                return 'i={}, data[0]={:.4f}'.format(self.i, data[0])

            except IndexError as e:
                self.active = False
                return e
            
            except Exception as e:
                return e

    def stop(self):
        self.view.playpause('pause')

if __name__ == '__main__':
    viewer = InstaCortexViewer(subj_id, 'rai2rfi', port=4567)
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