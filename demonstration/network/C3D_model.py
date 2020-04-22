import torch
import torch.nn as nn
from torchsummary import summary

        
class DilationBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_layer, step=1):
        super(DilationBlock, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.convlist = nn.ModuleList()
        for n in range(0, num_layer):
            dilation_t = n * step + 1
            padding_t = dilation_t
            module = nn.Sequential(nn.Conv3d(in_ch, 
                                              out_ch, 
                                              kernel_size=(3, 1, 1), 
                                              stride=(1, 1, 1), 
                                              padding=(0, 0, 0), 
                                              dilation=(dilation_t, 1, 1), 
                                              bias=False),
                                      nn.BatchNorm3d(out_ch),
                                      nn.ReLU(inplace=True), 
                                      )               
            self.convlist.append(module)

    def forward(self, x):
        output_list = []
        
        for layer in self.convlist:
            layer_output = layer(x)
            output_list.append(layer_output)
        
        outputs = torch.cat(output_list, dim=2)

        return outputs
        
        
            
class C3D_Dilation(nn.Module):
    def __init__(self, num_classes=24, pretrained=False):
        super(C3D_Dilation, self).__init__()
        self.dilation = DilationBlock(in_ch=3, out_ch=64, num_layer=3, step=2)
        self.norm1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))      
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))     
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4 = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))            
        
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5 = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.fc = nn.Linear(8192, 24)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = inputs
        
        x = self.dilation(x) # 64*48*112*112
        x = self.pool1(x) # 64*48*56*56
        
        x = self.conv2(x) # 128*48*56*56
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool2(x) # 128*24*28*28
        
        x = self.conv3(x) # 256*24*28*28
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool3(x) # 256*12*14*14
        
        x = self.conv4(x) # 512*12*14*14
        x = self.norm4(x)
        x = self.relu(x)
        x = self.pool4(x) # 512*6*7*7
        
        x = self.conv5(x) # 512*6*7*7
        x = self.norm5(x)
        x = self.relu(x)
        x = self.pool5(x) # 512*3*4*4 = 16384

        
        x = x.view(-1, 8192)
        x = self.fc(x)
        
        return x  
        
        
class C3D(nn.Module):
    def __init__(self, num_classes=24, pretrained=False):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.relu(self.conv5(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        logits = self.fc6(x)
        return logits

if __name__ == "__main__":
    model = C3D_Dilation().cuda()
    print(model)