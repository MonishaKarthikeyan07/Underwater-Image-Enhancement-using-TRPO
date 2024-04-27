# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from uwcc import uwcc
from model import PhysicalNN

def main():

    best_loss = 9999.0

    lr = 0.001
    batchsize = 1
    n_workers = 2
    epochs = 3000
    ori_fd = sys.argv[1]
    ucc_fd = sys.argv[2]
    ori_dirs = [os.path.join(ori_fd, f) for f in os.listdir(ori_fd)]
    ucc_dirs = [os.path.join(ucc_fd, f) for f in os.listdir(ucc_fd)]

    model = PhysicalNN()
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    trainset = uwcc(ori_dirs, ucc_dirs, train=True)
    trainloader = DataLoader(trainset, batchsize, shuffle=True, num_workers=n_workers)

    for epoch in range(epochs):
        tloss = train(trainloader, model, optimizer, criterion, epoch)
        print('Epoch:[{}/{}] Loss{}'.format(epoch, epochs, tloss))
        is_best = tloss < best_loss
        best_loss = min(tloss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Best Loss: ', best_loss)

def train(trainloader, model, optimizer, criterion, epoch):
    losses = AverageMeter()
    model.train()

    for i, sample in enumerate(trainloader):
        ori, ucc = sample
        ori = ori.cuda()
        ucc = ucc.cuda()

        corrected = model(ori)
        loss = criterion(corrected, ucc)
        losses.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg

def save_checkpoint(state, is_best):
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

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()