import torch
from torch.nn import Module, Conv1d, Linear, BatchNorm1d
from torch.functional import F

class EmotionsByGenderModel(Module):
    def __init__(self, n_class=14):
        super().__init__()
        self.conv1 = Conv1d(1, 256, kernel_size=7, padding=3)
        self.act1 = F.relu
        self.conv2 = Conv1d(256, 256, kernel_size=7, padding=3)
        self.bn1 = BatchNorm1d(256)
        self.act2 = F.relu
        self.dp1 = F.dropout
        self.pool1 = F.max_pool1d
        self.conv3 = Conv1d(256, 128, kernel_size=7, padding=3)
        self.act3 = F.relu
        self.conv4 = Conv1d(128, 128, kernel_size=7, padding=3)
        self.act4 = F.relu
        self.conv5 = Conv1d(128, 128, kernel_size=7, padding=3)
        self.act5 = F.relu
        self.conv6 = Conv1d(128, 128, kernel_size=7, padding=3)
        self.bn2 = BatchNorm1d(128)
        self.act6 = F.relu
        self.dp2 = F.dropout
        self.pool2 = F.max_pool1d
        self.conv7 = Conv1d(128, 64, kernel_size=7, padding=3)
        self.act7 = F.relu
        self.conv8 = Conv1d(64, 64, kernel_size=7, padding=3)
        self.act8 = F.relu
        self.conv9 = Conv1d(64, 64, kernel_size=7, padding=3)
        self.act9 = F.relu
        # 192 - 2.5 секунд
        # 384 - 5 секунд
        self.fc = Linear(384 + 1, n_class)
        self.act10 = F.softmax
    
    def forward(self, x):
        gender = x[0,0,-1]

        x = x[:,:,:-1]

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.act2(x)
        x = self.dp1(x, p=0.25)
        x = self.pool1(x, kernel_size=8)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.conv6(x)
        x = self.bn2(x)
        x = self.pool2(x, kernel_size=8)
        x = self.conv7(x)
        x = self.act7(x)
        x = self.conv8(x)
        x = self.act8(x)
        x = self.conv9(x)
        x = self.act9(x)

        x = x.flatten(start_dim=1, end_dim=2)
        
        g = torch.Tensor(x.shape[0] * [[gender]])

        if (torch.cuda.is_available()):
            g = g.cuda()

        x = torch.cat((x, g), dim=1)

        x = self.fc(x)

        x = self.act10(x, dim=1)

        return x