import os,glob,thread

# motion correction
for run in range(12):
    print 'starting mc run '+str(run+1)
    in_bold = glob.glob('./*run-'+str(run+1).zfill(3)+'*.nii')[0]
    out_bold = 'run-'+str(run+1).zfill(3)+'.nii'
    ref_bold = '../ref/rfi.nii'
    cmd = 'mcflirt -in '+in_bold+' -o '+out_bold+' -r '+ref_bold
    os.system(cmd)

# rename
for run in range(12):
    print 'starting rename run '+str(run+1)
    in_bold = glob.glob('./*run-'+str(run+1).zfill(3)+'*.nii')[0]
    out_bold = 'run-'+str(run+1).zfill(3)+'.nii'
    cmd = 'mv '+in_bold+' '+out_bold
    os.system(cmd)

# create rfi
in_bold = glob.glob('./*run-'+str(1).zfill(3)+'*.nii')[0]
out_bold = '../ref/rfi.nii'
os.system('fslmaths ' + in_bold + ' -Tmean ' + out_bold)