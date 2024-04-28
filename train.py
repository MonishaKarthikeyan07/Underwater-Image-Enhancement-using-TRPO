import torch
from trpo import TRPOAgent

def main():
    ori_dirs = ['/content/drive/MyDrive/Dataset/raw']
    ucc_dirs = ['/content/drive/MyDrive/Dataset/reference-890']
    batch_size = 32
    n_workers = 4
    epochs = 100

    trpo_agent = TRPOAgent()
    trpo_agent.train(ori_dirs, ucc_dirs, batch_size, n_workers, epochs)

if __name__ == '__main__':
    main()
