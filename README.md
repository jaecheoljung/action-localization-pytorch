# Action Localization

Step 1. Action recognition
 - Accuracy : ?
 
Step 2. Localization
 - To do.


## Training

Download [APS dataset](https://drive.google.com/file/d/1VFM1J2yem5L3m6Zabefv6Qveeh4DXnUj/view?usp=sharing)

Unzip dataset in root directory.

`python split_video.py aps_original aps_cut` 
will make a new folder named 'data' and put all video images there.
`python train.py`
will start training

if you want to test with unseen dataset, use `python test.py`
