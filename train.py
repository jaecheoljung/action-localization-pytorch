import timeit
from datetime import datetime
import socket
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

nEpochs = 100
snapshot = 50  # Store a model every snapshot epochs
lr = 1e-3
batch_size = 32
num_workers = 4
dataset = 'aps'
modelName = 'C3D'
num_classes = 24

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
run_dir = os.path.join(save_dir_root, 'run')
if not os.path.exists(run_dir):
	os.mkdir(run_dir)
save_dir = os.path.join(run_dir, 'run_' + str(run_id))
saveName = modelName + '-' + dataset

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr, num_epochs=nEpochs, save_epoch=snapshot):
    torch.cuda.empty_cache()
    ##model = C3D_model.C3D_Dilation(num_classes=num_classes, pretrained=False)
    model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    
    train_params = model.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Training {} from scratch...".format(modelName))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)
    #print(model)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=batch_size, num_workers=num_workers)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
        
            confusion_matrix = torch.zeros(num_classes, num_classes)
        
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                model.train()
                #scheduler.step()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device=device, dtype=torch.int64)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

            print(confusion_matrix.diag() / confusion_matrix.sum(1))

        if epoch % save_epoch == (save_epoch - 1):
            PATH = os.path.join(save_dir, saveName + '_epoch-' + str(epoch + 1) + '.pth.tar')
            torch.save(model.state_dict(), PATH)
            print("Save model at {}\n".format(PATH))

    writer.close()




if __name__ == "__main__":
    train_model()
