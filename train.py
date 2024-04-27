import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms

from model import PhysicalNN
from uwcc import uwcc
import shutil
import os
from torch.utils.data import DataLoader
import sys
from trpo import TRPO

def main():

    best_loss = 9999.0

    lr = 0.001
    batchsize = 1
    n_workers = 2
    epochs = 50
    ori_fd = sys.argv[1]
    ucc_fd = sys.argv[2]
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    # Create model
    model = PhysicalNN()
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load data
    trainset = uwcc(ori_dirs, ucc_dirs, train=True)
    trainloader = DataLoader(trainset, batchsize, shuffle=True, num_workers=n_workers)

    # Initialize TRPO
    trpo = TRPO(model, optimizer)

    # Train
    for epoch in range(epochs):
        tloss = trpo.train(trainloader)
        print('Epoch:[{}/{}] Loss{}'.format(epoch, epochs, tloss))
        is_best = tloss < best_loss
        best_loss = min(tloss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Best Loss: ', best_loss)

def save_checkpoint(state, is_best):
    """Saves checkpoint to disk"""
    freq = 500
    epoch = state['epoch']

    filename = './checkpoints/model_tmp.pth.tar'
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    torch.save(state, filename)

    if epoch % freq == 0:
        shutil.copyfile(filename, './checkpoints/model_{}.pth.tar'.format(epoch))
    if is_best:
        shutil.copyfile(filename, './checkpoints/model_best_{}.pth.tar'.format(epoch))

if __name__ == '__main__':
    main()
