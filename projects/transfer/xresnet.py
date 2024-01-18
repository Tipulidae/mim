from collections import OrderedDict

import torch
import torch.nn as nn

"""
Code adapted from https://github.com/tmehari/ssm_ecg
by Temesgen Mehari & Nils Strodthoff.
This is a refactored and cleaned up version with only the xresnet50
architecture.

Copyright stuff:
MIT License

Copyright (c) [2022] [Temesgen Mehari, Nils Strodthoff]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(1)
        self.mp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class ConvLayer(nn.Sequential):
    def __init__(
            self, ni, nf, ks=3, stride=1, bn_weight_init=1., act_cls=nn.ReLU,
            bias=False):

        layers = []
        conv = nn.Conv1d(
            ni, nf, kernel_size=ks, bias=bias, stride=stride,
            padding=((ks-1)//2)
        )
        layers.append(conv)

        bn = nn.BatchNorm1d(nf)
        bn.weight.data.fill_(bn_weight_init)
        layers.append(bn)

        if act_cls is not None:
            layers.append(act_cls())

        super().__init__(*layers)


def init_cnn(m):
    if getattr(m, 'bias', None) is not None:
        nn.init.constant_(m.bias, 0.)
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
    for child in m.children():
        init_cnn(child)


class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nf, stride=1):
        super().__init__()

        layers = [
            ConvLayer(ni*expansion, nf, ks=1),
            ConvLayer(nf, nf, ks=5, stride=stride),
            ConvLayer(nf, nf*expansion, ks=1, bn_weight_init=0., act_cls=None)
        ]
        self.convpath = nn.Sequential(nn.Sequential(*layers))
        idpath = []
        if stride != 1:
            idpath.append(nn.AvgPool1d(2, ceil_mode=True))
        if ni != nf:
            idpath.append(
                ConvLayer(ni * expansion, nf * expansion, 1, act_cls=None))

        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.convpath(x) + self.idpath(x))


class XResNet1d(nn.Sequential):
    def __init__(self, expansion, blocks, inp_dim=12, out_dim=1):
        self.expansion = expansion

        # Not sure if we really want bias for the first conv-layer, but
        # that's what the original does. It just happens in the most
        # roundabout way, and I'm not sure it was intentional.
        stem = [
            ConvLayer(inp_dim, 32, ks=5, stride=2, bias=True),
            ConvLayer(32, 32, ks=5, stride=1),
            ConvLayer(32, 64, ks=5, stride=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        ]

        res_segments = [
            self.make_res_segment(
                ni=64//expansion, size=blocks[0], initial_stride=1
            )
        ]
        res_segments.extend([
            self.make_res_segment(ni=64, size=block_size, initial_stride=2)
            for block_size in blocks[1:]
        ])

        head = nn.Sequential(
            OrderedDict(
                concat_pool=AdaptiveConcatPool1d(),
                flatten=nn.Flatten(),
                classifier=nn.Sequential(
                    nn.BatchNorm1d(2*64*expansion),
                    nn.Dropout(0.5),
                    nn.Linear(2*64*expansion, out_dim),
                ),
            )
        )

        # Using OrderedDict, we can name the various parts of the network
        # and later access (and modify) them as attributes of the model.
        # This makes the trained model much easier to use with transfer
        # learning. In my case, I particularly want to replace the 'classifier'
        # part of the 'head' during finetuning.
        super().__init__(
            OrderedDict(
                stem=nn.Sequential(*stem),
                res_segments=nn.Sequential(*res_segments),
                head=head
            )
        )
        init_cnn(self)

    def make_res_segment(self, ni, size, initial_stride):
        res_blocks = [ResBlock(self.expansion, ni, 64, stride=initial_stride)]
        for _ in range(size - 1):
            res_blocks.append(ResBlock(self.expansion, 64, 64, stride=1))

        return nn.Sequential(*res_blocks)
