""" Please refrain from changing any of the util code. """
from __future__ import division

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import argparse
import pprint
import os

import pandas as pd
import numpy as np

from PIL import Image
from scipy import misc
import sklearn.metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="densenet", type=str)
    parser.add_argument("--label_dir", default="data", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--seed", default=123456, type=int)
    parser.add_argument("--classes", default=52, type=int)
    parser.add_argument("--toy", dest="toy", action="store_true")
    parser.add_argument("--no-toy", dest="toy", action="store_false")
    parser.add_argument("--freeze", dest="fixed", action="store_true")
    parser.add_argument("--scratch",action="store_true")
    parser.add_argument("--save_path", default=None, type=str)

    return parser

def get_labels(label_dir="/deep/group/rad_data"):
    with open(os.path.join(label_dir, 'label2ind.txt')) as f:
        return f.read().splitlines()

def vec2labels(vec, label_dir="/deep/group/rad_data"):
    with open(os.path.join(label_dir, 'label2ind.txt')) as f:
        label2ind = {label: ind for ind, label in enumerate(f.read().splitlines())}
        ind2label = {ind: label for label, ind in label2ind.iteritems()}
    labels = [ind2label[ind] for ind, val in enumerate(vec) if val == 1 ]
    if len(labels) == 0:
        return "No Finding"
    else:
        return "|".join(labels)

class Dataset(data.Dataset):
    def __init__(self, label_dir, data_split,toy=False):
        super(Dataset, self).__init__()

        df = pd.read_csv(os.path.join(label_dir, "%s.csv" % data_split))
        if toy:
            df = df.sample(frac=0.01)

            # Remove any paths for which the image files do not exist
        df = df[df["Path"].apply(os.path.exists)]

        self.img_paths = df["Path"].tolist()

        # To convert from label vector to natural language labels,
        # use label2ind.txt (maps label to an index [line index])
        # the index of a label vector is 1 if that label appears, 0 otherwise
        self.labels = np.array([np.fromstring(row["Label"], sep= "|", dtype=int) for i, row in df.iterrows()])

        if data_split == "train":
            self.transform = transforms.Compose([
                                transforms.Scale(224),
                                transforms.ToTensor(),
                                #normalize,
            ])
        else:
            self.transform = transforms.Compose([
                                transforms.Scale(224),
                                transforms.ToTensor(),
                                #normalize,
            ])

    def __getitem__(self, index):
           img = Image.open(self.img_paths[index]).convert("RGB")
           label = self.labels[index]

           return self.transform(img), torch.LongTensor(label)

    def __len__(self):
        return len(self.img_paths)

def evaluate(gts, probabilities, use_only_index = None):
    assert(np.all(probabilities >= 0) == True)
    assert(np.all(probabilities <= 1) == True)

    def compute_metrics_for_class(i):
         AUC = sklearn.metrics.roc_auc_score(gts[:, i], probabilities[:, i])
         F1 = sklearn.metrics.f1_score(gts[:, i], preds[:, i])
         acc = sklearn.metrics.accuracy_score(gts[:, i], preds[:, i])
         count = np.sum(gts[:, i])
         return AUC, F1, acc, count

    AUCs = []
    F1s = []
    accs = []
    counts = []
    preds = probabilities >= 0.5
    labels = get_labels()

    classes = [use_only_index] if use_only_index is not None else range(len(gts[0]))
    for i in classes:
        try:
            AUC, F1, acc, count = compute_metrics_for_class(i)
        except ValueError:
            continue
        AUCs.append(AUC)
        F1s.append(F1)
        accs.append(acc)
        counts.append(count)
        print('Count: {:d} AUC: {:.4f} F1: {:.3f} Acc: {:.3f}'.format(count, AUC, F1, acc))
    avg_AUC = np.average(AUCs, weights=counts)
    print('Avg AUC: {:.3f}'.format(avg_AUC))
    avg_F1 = np.average(F1s, weights=counts)
    print('Avg F1: {:.3f}'.format(avg_F1))
    avg_acc = np.average(accs, weights=counts)
    print('Avg acc: {:.3f}'.format(avg_acc))
    
def loader_to_gts(data_loader):
    gts = []
    for (inputs, labels) in data_loader:
        for label in labels.cpu().numpy().tolist():
            gts.append(label)
    gts = np.array(gts)
    return gts

def load_data(args):
    train_dataset = Dataset(args.label_dir, "train", toy=args.toy)
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    test_dataset = Dataset(args.label_dir, "test", toy=args.toy)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    return train_loader, test_loader
