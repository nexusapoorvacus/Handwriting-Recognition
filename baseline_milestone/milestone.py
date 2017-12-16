from __future__ import division
from __future__ import print_function
import util
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

import pprint


class PreTrainedResNet(nn.Module):
    def __init__(self, config, nclasses):
        super(PreTrainedResNet, self).__init__()
        if config.scratch:
            self.model_ft = models.resnet18()
        else:
            self.model_ft = models.resnet18(pretrained=True)
        if config.fixed:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, config.classes)
        self.config = config

    def forward(self, x):
        return self.model_ft(x)

class DenseNet(nn.Module):
    def __init__(self, config, nclasses):
        super(DenseNet, self).__init__()
        if config.scratch:
            self.model_ft = models.densenet121(pretrained=True)
        else:
            self.model_ft = models.densenet121()
        if config.fixed:
            for param in self.model_ft.parameters():
                param.requires_grad = False
        num_ftrs = self.model_ft.classifier.in_features
        self.model_ft.classifier = nn.Linear(num_ftrs, nclasses)
        self.config = config
    def forward(self, x):
        return self.model_ft(x)



def transform_data(data, use_gpu):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad = False)
    labels = Variable(labels)
    return inputs, labels


def train_epoch(epoch, args, model, loader, criterion, optimizer):
    model.train()
    test_interval = args.workers*3 - 1
    for batch_idx, data in enumerate(loader):
        inputs, labels = transform_data(data, True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print("Epoch: {:d} Batch: {:d} ({:d}) Train Loss: {:.6f}".format(
            epoch, batch_idx, args.batch_size, loss.data[0]))        
        sys.stdout.flush()
    return loss.data[0]  

def test_epoch(model, loader, criterion):
    model.eval()
    test_losses = []
    outs = []
    gts = []
    for data in loader:
        for label in data[1].numpy().tolist():
            gts.append(label)
        inputs, labels = transform_data(data, True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_losses.append(loss.data[0])
        out = torch.sigmoid(outputs).data.cpu().numpy()
        outs.extend(out)
    avg_loss = np.mean(test_losses)
    print('Validation Loss: {:.6f}'.format(avg_loss))

    outs = np.array(outs)
    gts = np.array(gts)
    util.evaluate(gts, outs)
    return avg_loss

def run(args):
    train_loader, test_loader = util.load_data(args)
    use_gpu = torch.cuda.is_available()

    if args.model == "resnet":
        model = PreTrainedResNet(args, args.classes)
    elif args.model == "densenet":
        model = DenseNet(args, args.classes)
    else:
        print("{} is not a valid model.".format(args.model))

    if use_gpu:
        model = model.cuda()

    criterion = nn.MultiLabelSoftMarginLoss()
    if args.optimizer == 'adam':
        optimizer  = optim.Adam(
                        filter(lambda p: p.requires_grad, model.model_ft.parameters()),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    best_model_wts, best_loss = model.state_dict(), float("inf")
    for epoch in range(1, args.epochs + 1):
        print('Epoch {}/{}'.format(epoch, args.epochs))
        print('-' * 10)
        train_loss = train_epoch(epoch, args, model, train_loader,criterion, optimizer)
        scheduler.step()
        print("Testing")
        test_loss = test_epoch(model, test_loader, criterion)    
        if (test_loss < best_loss):
            best_loss = test_loss
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, args.save_path)
            #torch.save(best_model_wts, os.path.join(args.save_path, "test%f_train%f_epoch%d" % (test_loss, train_loss, epoch)))

if __name__ == "__main__":
    parser = util.get_parser()
    args = parser.parse_args()
    pp = pprint.PrettyPrinter()
    pp.pprint(vars(args))
    run(args)
