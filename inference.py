import torch
import numpy as np
from network import C3D_model
import cv2
torch.backends.cudnn.benchmark = True

weight = 'run/run_0/models/C3D-aps_epoch-49.pth.tar'
video = './aps_original/gesture18/g18s20.avi' # 0 if using webcam

# 128x171
def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/aps_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    model = C3D_model.C3D(num_classes=24)
    checkpoint = torch.load(weight, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video)
    retaining = True
    wait = False
    
    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            
            probs = probs[0][label]
            name = class_names[label].split(' ')[-1].strip()
            
            cv2.putText(frame, name, (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            clip.pop(0)
            
            if wait == True and probs < 0.4:
                wait = False
                
            if wait == False and probs > 0.9:
                print(name)
                wait = True

        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









