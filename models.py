import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, cnn_cfg, cnn_top, nclasses):
        super(CNN, self).__init__()

        in_channels = 8
        self.features = nn.ModuleList([nn.Conv2d(1, in_channels, 7, 2, 3), nn.ReLU()])

        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x))
                    in_channels = x
                    cnt += 1

        input_size = in_channels
        hidden_size = cnn_top
        self.temporal = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.1),
            nn.Conv2d(hidden_size, nclasses, kernel_size=(1, 5), stride=1, padding=(0, 2)),
        )

        self.scale = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.1),
            nn.Conv2d(hidden_size, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, bbox=None, reduce=True, scale=True):

        y = x
        if bbox is not None:
            bbox = bbox.to(x.device)

        for nn_module in self.features:
            y = nn_module(y)

        s = 1
        if scale:
            s = F.sigmoid(self.scale(y))

        if reduce:
            yc = self.temporal(y)

            if bbox is not None:
                bbox_mask = (F.interpolate(bbox, size=[y.size(2), y.size(3)]) > .1).float() #.clamp(.1, 1)
                bbox_mask_aux = bbox_mask
                if self.training:
                     bbox_mask_aux *= torch.bernoulli(.95 * torch.ones_like(bbox_mask_aux))

                bbox_mask_aux =  bbox_mask_aux.repeat(1, yc.size(1), 1, 1)
                bbox_mask_max = -10. * yc.abs().mean() * (1 - bbox_mask_aux).detach()

                yctc = F.max_pool2d(yc + bbox_mask_max.detach(), [y.size(2), 1], stride=[y.size(2), 1], padding=[0, 0])
                ycnt = (F.softmax(yc, 1) * s * bbox_mask.detach())[:, 1:].sum(2).sum(2)

                char_mask = (F.softmax(yc, 1)[:, 0] > .5).float().detach()
                eloss = .1 * torch.sum(char_mask * s) / torch.sum(char_mask)
            else:
                yctc = F.max_pool2d(yc, [y.size(2), 1], stride=[y.size(2), 1], padding=[0, 0])
                ycnt = (F.softmax(yc, 1) * s)[:, 1:].sum(2).sum(2)

                eloss = 0

            if self.training:
                len_in = y.size(0) * [y.size(3)]
                return ycnt, len_in, yctc.permute(2, 3, 0, 1)[0], eloss
            else:
                return ycnt, yctc.permute(2, 3, 0, 1)[0]
        else:
            yc = self.temporal(y)
            return yc, s