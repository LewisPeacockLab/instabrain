%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% datastructure hierarchy and callbacks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% img_file = event.src_path.rsplit('/')[-1]
% if img_file.endswith('imgdat'):
%     if img_file.startswith('MB'):
%         self.sp_pool.apply_async(func = proc_epi_slice,
%                                  args = (img_file, event.src_path),
%                                  callback = self.sp.add_to_proc_epi_dict)

% def proc_epi_slice(img_file, path):
%     rep = int(img_file.split('R')[1].split('-')[0])
%     slc = int(img_file.split('S')[1].split('.')[0])
%     with open(path) as f:
%         slice_data = np.fromfile(f,dtype=np.uint16)
%     return rep, slc, slice_data

% def add_to_proc_epi_dict(self, (rep,slc,slice_data)):
%     if rep not in self.vol_status_dict:
%         self.vol_status_dict[rep] = 0
%     if rep not in self.proc_vol_dict:
%         self.proc_vol_dict[rep] = np.zeros(self.epi_dims, dtype=np.uint16)
%     self.vol_status_dict[rep] += 1
%     self.proc_vol_dict[rep][int(slc)-1] = slice_data
%     if self.vol_status_dict[rep] == self.epi_slices:
%         self.proc_epi_volume(rep)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% watch folder
% when new slice (or volume) object appears...
% --> do processing
% then output classifier result to frontend

% realtime steps:
% import slice by slice (or volume) data
% motion correction
% (smoothing)
% z-scoring
% linear detrend
% moving average
% apply classifier 
% 
