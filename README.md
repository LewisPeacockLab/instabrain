# instabrain
Instabrain is a python tool for decoding fMRI patterns in real-time. This processing includes (1) motion correction using existing packages such as AFNI or FSL, (2) time series processing of voxel activities using NumPy and SciPy, and (3) decoding using PyMVPA classifiers.

# Installing
For now, just install dependencies. Some day, this will be a package.

python 2.7
```python
pip install -r requirements.txt
```
python 3
```python
pip3 install -r requirements.txt
```

# Operation

## Display

The `instabrain/display` code is a sample PsychoPy program to display decoded neurofeedback (e.g. [Shibata et al. (2011)](https://www.ncbi.nlm.nih.gov/pubmed/22158821)). The main `game.py` script is configured using `game_config.yml`. A basic flask server (`game_server.py`) is provided to accept feedback data over the network, but any HTTP server can be used to accept data and pass it to your own custom display script.

## Localizer

Instabrain uses PyMVPA classifiers by default. The sample `localizer/insta_localizer.py` script trains a PyMVPA classifier that is saved into a python-readable pickled file called `clf.p`. Any PyMVPA classifier object, created using your own custom script, can be pickled for use in real-time feedback.

## Realtime

The main feedback calculation script is `realtime/instabrain.py`. This script must be run with __reference data__ (`-s subject_id`) and a __configuration file__ (`-c study_name`). There are also optional flags for __debugging with pre-recorded data__ (`-d`) and __logging performance__ (`-l`).

### Reference data

There are two required reference files: a reference functional (EPI) image (`realtime/ref/subject_id/rfi.nii`) and a PyMVPA classifier (`realtime/ref/subject_id/clf.p`). The reference functional image must be in the same resolution as the incoming realtime data. `rfi.nii` should be a single volume, or averaged volume from the localizer session, and all images used to train `clf.p` should be motion corrected to `rfi.nii`.

A third, optional reference file is `class.txt`. This can be used by the front-end `display/game.py` to determine which of the classifier outputs to use as the feedback value. For example, in decoded neurofeedback experiment, feedback should be provided from only one of the classifier outputs. In [Shibata et al. (2011)](https://www.ncbi.nlm.nih.gov/pubmed/22158821), each participant received feedback from one of three (randomly chosen) target stimuli. The file `class.txt` should contain an integer value of which of the `n` classes to use as feedback (e.g., if there are 3 classes, it must contain a number from 1 to 3). If this file is not provided, the `realtime/instabrain.py` script will supply `-1` as the target class, and it is up to you (in your custom display script) to determine how you want to calculate your displayed feedback value from the `n` provided classifier outputs.

### Configuration file

The configuration file should be named after the study being performed. A sample configuration is provided in `realtime/config/default.yml`. For each study, a new `config/my_study_name.yml` file should be created.

A rundown of the configuration parameters is as follows:

#### file processing
`watch-dir`: `path/to/directory` of the directory where incoming images will appear

`recon-script`: `path/to/script.sh` of the path to the remote reconstruction script

`archive-data`: `True` or `False` value or whether or not to save realtime data. Should only be set to `True` during benchmarking and debugging

#### networking
`post-url`: `http://XXX.XXX.XXX.XXX:PORT/post_directory` to send realtime data via POST requests

#### imaging params
`baseline-trs`: `integer` of the number of TRs to use for baseline calculation

`feedback-trs`: `integer` containing number of TRs per run, not including baseline TRs

#### data processing

`moving-avg-trs`: `integer` containing number of TRs to use for moving average

`mc-mode`: `string` containing either `AFNI` or `FSL` used to determined which motion correction will be used (AFNI's `3dvolreg` or FSL's `mcflirt`)

#### MRI sequence
`multiband`: `True` or `False` value indicating whether the sequence is multiband or not

### debugging
The `-d` flag makes the watcher look for real-time data in the `instabrain/data` folder instead of from the MRI scanner. Only use this flag if you are testing instabrain away from the scanner.

The `-l` flag indicates whether performance logs should be recorded. Only use this flag if you are benchmarking performance.

### Running the `realtime/instabrain.py` script

The `instabrain.py` script should be run from within the realtime folder. The `/proc` folder is used for storing temporary files during motion correction.

Running `instabrain.py` is simple as long as the required python packages have been installed and a motion correction script (either AFNI's `3dvolreg` or FSL's `mcflirt`) is in the system's `PATH` variable. For example, if a subject `bert` has been scanned in a localizer session, the files `realtime/ref/bert/rfi.nii` and `realtime/ref/bert/clf.p` must be created from this session. If we wish to run `bert` in a study called `cool_neurofeedback_study`, we must also create a `realtime/config/cool_neurofeedback_study.yml` containing the custom configuration for our experiment. Then, the experiment can be run using `python instabrain.py -s bert -c cool_neurofeedback_study`. Add the debugging or logging booleans (e.g. `python instabrain.py -s subjid -c myconfig -d -l`) to enable those functionalities.

Some day, a Docker implementation will be provided. This will ease the AFNI/FSL requirements by automatically including them in the Docker image. You will have to create your own `ref` and `config` folders wherever you like on your host machine, including the data of your participants and studies. Then, you will provide these folders, as well as your subject ID and study name to a custom `instabrain.sh` script, which will run instabrain as normal.
