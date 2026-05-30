import argparse
from itertools import product
import torch
import torch.nn as nn
import numpy as np
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
import sys
import os
import pandas as pd

sys.path.append(os.path.join(sys.path[0], '..'))

from utils import get_model
from data import get_cifair10_datasets
from perturb import (
    gaussian_noise_dataset,
    blur_dataset,
    brightness_dataset,
    random_erasure_dataset,
    jpeg_compression_dataset,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on CIFAR-10')
parser.add_argument('--batch_size', metavar='BSZ', type=int, help='batch size', default=128)
parser.add_argument('--num_workers', metavar='NW', type=int, help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', type=int, help='number of calibration points', default=10000)
parser.add_argument('--num_train_head', metavar='NTRAINHEAD', type=int, help='number of points used to train classification head', default=10000)
parser.add_argument('--head_epochs', metavar='HEADEPOCHS', type=int, help='epochs to train the 10-class output layer', default=10)
parser.add_argument('--head_lr', metavar='HEADLR', type=float, help='learning rate for head-only training', default=1e-2)
parser.add_argument('--seed', metavar='SEED', type=int, help='random seed', default=0)


def _extract_logits(output):
    if isinstance(output, tuple):
        first = output[0]
        if isinstance(first, tuple):
            return first[0]
        return first
    if hasattr(output, 'logits'):
        return output.logits
    return output


def adapt_output_layer_to_10_classes(model, modelname):
    module = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Freeze all parameters; only the replaced output layer will be trainable.
    for param in module.parameters():
        param.requires_grad = False

    if modelname in ['ResNet18', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt101', 'ShuffleNet', 'Inception']:
        in_features = module.fc.in_features
        module.fc = nn.Linear(in_features, 10).cuda()
        trainable_params = module.fc.parameters()
    elif modelname == 'DenseNet161':
        in_features = module.classifier.in_features
        module.classifier = nn.Linear(in_features, 10).cuda()
        trainable_params = module.classifier.parameters()
    elif modelname == 'VGG16':
        in_features = module.classifier[6].in_features
        module.classifier[6] = nn.Linear(in_features, 10).cuda()
        trainable_params = module.classifier[6].parameters()
    else:
        raise NotImplementedError(f'Unsupported model for head replacement: {modelname}')

    for param in trainable_params:
        param.requires_grad = True


def train_output_layer(model, train_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss().cuda()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_seen = 0
        for x, targets in train_loader:
            x = x.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            optimizer.zero_grad()
            logits = _extract_logits(model(x))
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.shape[0]
            n_seen += x.shape[0]

        avg_loss = running_loss / max(n_seen, 1)
        print(f'  Head epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}')
    model.eval()

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)


    # Get ciFAIR-10 data and split into head-training and conformal sets.
    cifair_train_data, _ = get_cifair10_datasets(download=True)
    if args.num_train_head + args.num_calib > len(cifair_train_data):
        raise ValueError('num_train_head + num_calib must be <= number of ciFAIR-10 training samples.')

    n_val = len(cifair_train_data) - args.num_train_head - args.num_calib
    cifair_head_train_data, cifair_calib_data, cifair_val_data = torch.utils.data.random_split(
        cifair_train_data,
        [args.num_train_head, args.num_calib, n_val],
    )

    # Initialize loaders 
    head_train_loader = torch.utils.data.DataLoader(cifair_head_train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    calib_loader = torch.utils.data.DataLoader(cifair_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    cudnn.benchmark = True

    # Get the models
    modelnames = ['ResNeXt101','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
    alphas = [0.05, 0.10]
    predictors = ['Fixed','Naive', 'APS', 'RAPS']

    perturbations = {
        'clean': cifair_val_data,
        'gaussian_noise': gaussian_noise_dataset(cifair_val_data, std=0.1),
        'blur': blur_dataset(cifair_val_data, kernel_size=5, sigma=1.5),
        'brightness': brightness_dataset(cifair_val_data, brightness_factor=0.5),
        'erasure': random_erasure_dataset(cifair_val_data, scale=(0.1, 0.3)),
        'jpeg': jpeg_compression_dataset(cifair_val_data, quality=10),
    }

    perturb_loaders = {name: torch.utils.data.DataLoader(perturb_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=args.num_workers) 
                        for name, perturb_dataset in perturbations.items()}
    
    cols = ["Model","Predictor","Alpha","Perturbation","Top1","Top5","Coverage","Size"]
    results = pd.DataFrame(columns = cols)

    for modelname in modelnames:
        # Load pretrained backbone and adapt only the output layer to 10 classes.
        base_model = get_model(modelname).cuda()
        adapt_output_layer_to_10_classes(base_model, modelname)
        print(f'Training 10-class output layer for {modelname}...')
        train_output_layer(base_model, head_train_loader, args.head_epochs, args.head_lr)
        base_model.eval()

        # optimize for 'size' or 'adaptiveness'
        lamda_criterion = 'size'
        # allow sets of size zero
        allow_zero_sets = False 
        # use the randomized version of conformal
        randomized = True 


        for alpha, predictor in product(alphas, predictors):
            print(f'Conformalizing {modelname} with alpha={alpha} and predictor={predictor}...')

            ### Experiment logic
            if predictor in ['Fixed', 'Naive', 'APS']:
                kreg = 1
                lamda = 0 # No regularization.
            else:
                kreg = None
                lamda = None

            # Conformalize adapted 10-class model
            conformal_model = ConformalModel(base_model,
                                             calib_loader,
                                             alpha=alpha,
                                             kreg=kreg,
                                             lamda=lamda,
                                             randomized=randomized,
                                             allow_zero_sets=allow_zero_sets)

            print("Model calibrated and conformalized! Now evaluate over remaining data.")

            for perturb_name, perturb_loader in perturb_loaders.items():
                print(f'Evaluating on perturbation: {perturb_name}...')
                
                top1_avg, top5_avg, coverage_avg, size_avg = validate(perturb_loader, conformal_model, print_bool=False)

                results = pd.concat([results,
                                     pd.DataFrame([[modelname,
                                                    predictor,
                                                    alpha,
                                                    perturb_name,
                                                    top1_avg,
                                                    top5_avg,
                                                    coverage_avg,
                                                    size_avg]],
                                                  columns=cols)],
                                    ignore_index=True)

            print(results)
        
    print("Complete!")
    results.to_csv('results.csv', index=False)