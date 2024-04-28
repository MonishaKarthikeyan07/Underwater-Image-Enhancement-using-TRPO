import argparse
from trpo import TRPOAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dir', help='Directory containing original images', required=True)
    parser.add_argument('--ucc_dir', help='Directory containing transformed images', required=True)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    args = parser.parse_args()

    ori_dirs = [args.ori_dir]
    ucc_dirs = [args.ucc_dir]

    trpo_agent = TRPOAgent()
    trpo_agent.train(ori_dirs, ucc_dirs, args.batch_size, args.n_workers, args.epochs)

if __name__ == '__main__':
    main()
