import torch
from torch.utils.data import DataLoader
from uwcc import uwcc
from model import PhysicalNN
import os

class TRPOAgent:
    def __init__(self):
        self.policy = PhysicalNN()  # Define your policy network
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

    def collect_samples(self, ori_dirs, ucc_dirs, batch_size, n_workers):
        train_set = uwcc(ori_dirs, ucc_dirs, train=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        return train_loader

    def compute_advantages(self, rewards):
        # This is a simple example, you may need to implement a more sophisticated advantage estimation method
        return rewards - rewards.mean()

    def train(self, ori_dirs, ucc_dirs, batch_size, n_workers, epochs):
        dataloader = self.collect_samples(ori_dirs, ucc_dirs, batch_size, n_workers)

        best_loss = 9999.0
        for epoch in range(epochs):
            losses = self.train_epoch(dataloader)
            tloss = losses.avg

            print('Epoch:[{}/{}] Loss{}'.format(epoch, epochs, tloss))
            is_best = tloss < best_loss
            best_loss = min(tloss, best_loss)

            # Save checkpoint
            if is_best:
                self.save_checkpoint(epoch)

        print('Best Loss: ', best_loss)

    def train_epoch(self, dataloader):
        losses = AverageMeter()
        self.policy.train()

        for i, sample in enumerate(dataloader):
            ori, ucc = sample

            corrected = self.policy(ori)
            loss = self.criterion(corrected, ucc)
            losses.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return losses

    def save_checkpoint(self, epoch):
        filename = './checkpoints/model_tmp.pth.tar'
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filename)
