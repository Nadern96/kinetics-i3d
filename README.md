# I3D models trained on Kinetics

## Overview

Original implementation by the authors can be found in this [repository](https://github.com/deepmind/kinetics-i3d), together with details about the pre-processing techniques.


## Running the code

### Setup
clone this repository using

$ git clone https://github.com/Nadern96/kinetics-i3d

# Requirements

```
pip install -r requirements.txt
```
Sonnet](https://github.com/deepmind/sonnet).


### Sample code

Run the example code using

`$ python evaluate_sample.py`

With default flags, this builds the I3D two-stream model, loads pre-trained I3D
checkpoints into the TensorFlow session, and then passes an example video
through the model. The example video has been preprocessed, with RGB
NumPy arrays provided (see more details below).

The script outputs the top 5 Kinetics classes predicted by the model 
with their probability. Using the default flags, the output should 
resemble the following up to differences in numerical precision:

```
Norm of logits: 76.697174

Top 5 classes and probabilities
playing cricket        100.00%
shooting goal (soccer) 0.00%
hurling (sport)        0.00%
catching or throwing softball 0.00%
catching or throwing baseball 0.00%
```

### Running the test

The test file can be run using

`$ python i3d_test.py`

This checks that the model can be built correctly and produces correct shapes.

## Further details

### Provided checkpoints

The default model has been pre-trained on ImageNet and then Kinetics; other
flags allow for loading a model pre-trained only on Kinetics and for selecting
only the RGB or Flow stream. The script `multi_evaluate.sh` shows how to run all
these combinations, generating the sample output in the `out/` directory.

The directory `data/checkpoints` contains the four checkpoints that were
trained. The ones just trained on Kinetics are initialized using the default
Sonnet / TensorFlow initializers, while the ones pre-trained on ImageNet are
initialized by bootstrapping the filters from a 2D Inception-v1 model into 3D,
as described in the paper.

The models are trained using the training split of Kinetics. On the Kinetics
test set, we obtain the following top-1 / top-5 accuracy:

Model          | ImageNet + Kinetics | Kinetics
-------------- | :-----------------: | -----------
RGB-I3D        | 71.1 / 89.3         | 68.4 / 88.0
Flow-I3D       | 63.4 / 84.9         | 61.5 / 83.4
Two-Stream I3D | 74.2 / 91.3         | 71.6 / 90.0

# Sample data and preprocessing

## preprocessing

The preprocessing file can be run using

`$ python preprocessing.py --video_path "Path_to the Video"

Ex: 

  python preprocessing.py --video_path "VID_NaderSquat.mp4"
 `
This will convert the video to a numpy array in the data/ directory 
with the name "VID_NaderSquat_rgb.npy"

## Data

Under data/ there are 3 input videos.
- cricket.avi
- VID_NaderRun.mp4
- VID_MultiAction.mp4

And also their numpypy arrays.

## Test with your own video
`$ python preprocessing.py --video_path "Path_to the Video'

replace _SAMPLE_PATHS in evaluate_sample.py with the path 
of your video numpyarray

Then run evaluate_Sample.py
$ python preprocessing.py
` 
 
