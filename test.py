import timeit
from datetime import datetime
import os
import glob
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from dataloaders.dataset import VideoDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from network import C3D_model

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

batch_size = 16
num_workers = 4
dataset = 'aps'
modelName = 'C3D'
num_classes = 24

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
run_id = int(runs[-1].split('_')[-1])
save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
saveName = modelName + '-' + dataset

def test_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, num_epochs=nEpochs):
    torch.cuda.empty_cache()
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(os.path.listdir(save_dir)[-1]))
    params = model.parameters()

    criterion = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=batch_size, num_workers=num_workers)
    test_size = len(test_dataloader.dataset)

    model.to(device)
    criterion.to(device)
    #print(model)

    model.eval()
    start_time = timeit.default_timer()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / test_size
    acc = running_corrects.double() / test_size

    print("[test] Loss: {} Acc: {}".format(loss, acc)
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")



if __name__ == "__main__":
    test_model()