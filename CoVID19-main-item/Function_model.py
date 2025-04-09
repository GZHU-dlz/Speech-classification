import math, random
import torch
import torch.nn as nn

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Fsq,squeeze过程
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),  # 两次全连接，Excitation
            nn.Sigmoid(),  # 限制到[0,1]
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class SEModule22(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule22, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out

class Tiss(nn.Module):
    def __init__(self, m=-0.80):
        super(Tiss, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class TASS(nn.Module):
    def __init__(self, channel):
        super(TASS, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        t = 3
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv1d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Tiss()
        self.attentions=Simam_module()

    def forward(self, input,insum):
        x = self.avg_pool(input) #[64,1]
        x_mfcc = self.avg_pool(insum)
        input=input.unsqueeze(0)
        insum=insum.unsqueeze(0)
        input_new = self.attentions(input)
        insum_new = self.attentions(insum)
        input_new = input_new.squeeze(0)
        insum_new = insum_new.squeeze(0)
        x1 = self.conv1(x)
        x_mfcc1 = self.conv1(x_mfcc)
        x2 = self.fc(x).transpose(-1, -2)
        x_mfcc2 = self.fc(x_mfcc1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x_mfcc2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x_mfcc1.transpose(-1, -2), x2.transpose(-1, -2)), dim=1).unsqueeze(
                -1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1))
        out = self.sigmoid(out)
        input_insum = (input_new * 0.6) + (insum_new * 0.4)

        return input_insum * out

class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, h, w = x.size()
        x_order=x
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[1, 2], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[1, 2], keepdim=True) / n + self.e_lambda)) + 0.5
        return x_order+x * self.act(y)

class ECAPA_TDNN(nn.Module):

    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()

        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.conv1_mfcc3=nn.Conv1d(in_channels=24,out_channels=12,kernel_size=1)
        self.conv1_mfcc2=nn.Conv1d(in_channels=128,out_channels=64,kernel_size=1)
        self.conv1_mfcc1=nn.Conv1d(in_channels=80,out_channels=40,kernel_size=1)
        self.conv1_mfcc = nn.Conv1d(40, C, kernel_size=5, stride=1, padding=2)
        self.relu_mfcc = nn.ReLU()
        self.bn1_mfcc = nn.BatchNorm1d(C)
        self.layer1_mfcc = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2_mfcc = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3_mfcc = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4_mfcc = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention_mfcc = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn7 = nn.BatchNorm1d(3072)
        self.fc8 = nn.Linear(3072, 2)
        self.bn8 = nn.BatchNorm1d(2)
        self.miss=TASS(64)

    def forward(self, x,x_mel):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        t = x.size()[-1]
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)

        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn7(x)
        x = self.fc8(x)
        x = self.bn8(x)
        x_mel = self.conv1_mfcc(x_mel)
        x_mel = self.relu_mfcc(x_mel)
        x_mel = self.bn1_mfcc(x_mel)
        x1_mel = self.layer1_mfcc(x_mel)
        x2_mel = self.layer2_mfcc(x_mel + x1_mel)
        x3_mel = self.layer3_mfcc(x_mel + x1_mel + x2_mel)
        x_mel = self.layer4_mfcc(torch.cat((x1_mel, x2_mel, x3_mel), dim=1))
        x_mel = self.relu_mfcc(x_mel)
        t_mel = x_mel.size()[-1]
        global_x_mel = torch.cat((x_mel, torch.mean(x_mel, dim=2, keepdim=True).repeat(1, 1, t_mel),
                                   torch.sqrt(torch.var(x_mel, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1,
                                                                                                             t_mel)),
                                  dim=1)

        w_mel = self.attention(global_x_mel)
        mu_mel = torch.sum(x_mel * w_mel, dim=2)
        sg_mel = torch.sqrt((torch.sum((x_mel ** 2) * w_mel, dim=2) - mu_mel ** 2).clamp(min=1e-4))
        x_mel = torch.cat((mu_mel, sg_mel), 1)
        x_mel = self.bn7(x_mel)
        x_mel = self.fc8(x_mel)
        x_mel = self.bn8(x_mel)
        x=self.miss(x,x_mel)
        return x, self.fc8
