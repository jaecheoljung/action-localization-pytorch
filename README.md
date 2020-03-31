# Action Localization

Step 1. Action recognition (modified. original code source: https://github.com/jfzhang95/pytorch-video-recognition)
 - Accuracy
 Train accuracy over 98%
 Validation accuracy over 92%
 
 
Step 2. Localization
 - Using YOLO v3 (modified. original code source: https://github.com/ayooshkathuria/pytorch-yolo-v3)

<table style="border:0px">
   <tr>
       <td><img src="yolov3/video_output/demo.mp4" frame=void rules=none></td>
   </tr>
</table>

## Classification Training

Download [APS dataset](https://drive.google.com/file/d/1VFM1J2yem5L3m6Zabefv6Qveeh4DXnUj/view?usp=sharing)

Unzip dataset in root directory.

`python split_video.py aps_original aps_cut` 
will make a new folder named 'aps_cut' and put all split videos there.

`python train.py`
will make a new folder named 'data' and start training.

Training Results are like below.

![Training Result](https://slack-files.com/TT1D7UWE6-F0113UGS2CQ-103250c6d8)

### YOLO v3
Go to yolov3 directory.

Download pre-trained weight [here](https://pjreddie.com/media/files/yolov3.weights)

`python video.py --video video_input/aps.avi video_output/`
