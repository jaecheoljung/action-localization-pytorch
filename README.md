# Action Localization

Step 1. Action recognition (modified. original code source: https://github.com/jfzhang95/pytorch-video-recognition)
 - Accuracy
 Train accuracy over 98%
 Validation accuracy over 92%
 
 
Step 2. Localization
 - Using YOLO v3 (modified. original code source: https://github.com/ayooshkathuria/pytorch-yolo-v3)
 
 ![Localization Example](https://jay-jro5362.slack.com/files/UTE6ZM93J/F0113SSCKDK/output_9.mp4)


## Classification Training

Download [APS dataset](https://drive.google.com/file/d/1VFM1J2yem5L3m6Zabefv6Qveeh4DXnUj/view?usp=sharing)

Unzip dataset in root directory.

`python split_video.py aps_original aps_cut` 
will make a new folder named 'aps_cut' and put all split videos there.

`python train.py`
will make a new folder named 'data' and start training.

Training Results are like below.

![Training Result](https://jay-jro5362.slack.com/files/UTE6ZM93J/F0113UGS2CQ/base-6-9-16-19.jpg)

### YOLO v3
Go to yolov3 directory.

Download pre-trained weight [here](https://pjreddie.com/media/files/yolov3.weights)

`python video.py --video video_input/aps.avi video_output/`
