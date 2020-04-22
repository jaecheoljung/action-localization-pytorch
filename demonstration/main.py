from __future__ import division
import time
import torch 
import torch.nn as nn
import numpy as np
import cv2
import os
import pickle as pkl
from network.util import *
from cv2 import VideoWriter, VideoWriter_fourcc
from network.darknet import Darknet
from network.C3D_model import C3D, C3D_Dilation


CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Darknet("network/yolov3.cfg")
model.load_weights("network/yolov3.weights")
model.net_info["height"] = 416
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

model2 = C3D(num_classes=24)
model2.load_state_dict(torch.load("network/C3D.pth"))

model.to(device)
model2.to(device)

model.eval()
model2.eval()


with open('network/label.txt', 'r') as f:
    class_names = f.readlines()
    f.close()

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label == "person":
        label = name
    color = colors[cls % len(colors)]
    cv2.rectangle(img, c1, c2, color, 2)
    
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] - t_size[1] - 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1]), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

videofile = 'input.avi'
cap = cv2.VideoCapture(videofile)  
assert cap.isOpened(), 'Cannot capture source'

frames = 0
start = time.time()

fourcc = VideoWriter_fourcc(*'mp4v')
video_writer = VideoWriter("output.mp4", fourcc, 20, (320, 240))

clip = []
cnt = 0
name = "undefined"

confidence = 0.5
nms_thresh = 0.4
num_classes = 80

while cap.isOpened():
    ret, frame = cap.read()
    cnt += 1    
    if ret:
        img = prep_image(frame, inp_dim) # output: torch tensor
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)
        
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(img, CUDA)

        if cnt == 4:
            cnt = 0
            tmp_ = center_crop(cv2.resize(frame, (171, 128)))
            tmp = tmp_ - np.array([[[112.0, 105.0, 103.0]]])
            clip.append(tmp)

        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model2.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            probs = probs[0][label]
            print("label: {}, probs: {}".format(label, probs))
            name = class_names[label].split(' ')[-1].strip()
            clip.pop(0)
            
        output = write_results(output, confidence, num_classes, nms_conf = nms_thresh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            #cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        classes = load_classes('network/coco.names')
        colors = pkl.load(open("network/pallete", "rb"))

        list(map(lambda x: write(x, frame), output))
        
        #cv2.imshow("frame", frame)
        video_writer.write(frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        #print(time.time() - start)
        #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     

video_writer.release()