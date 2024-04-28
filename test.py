import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime
from trpo import TRPO

def main(checkpoint, imgs_path, result_path):

    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    # TRPO instance
    trpo = TRPO(model)

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    starttime = datetime.datetime.now()
    for imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(imgdir))[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.to(device)
        out = trpo.predict(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = os.path.join(result_path, 'results_{}'.format(checkpoint['epoch']))
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(os.path.join(dir, '{}_corrected.png'.format(img_name)))
    endtime = datetime.datetime.now()
    print("Total time taken:", endtime - starttime)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoints path', required=True)
    parser.add_argument('--images', help='test images folder', default='./test_img/')
    parser.add_argument('--result', help='results folder', default='./results/')
    args = parser.parse_args()
    main(checkpoint=args.checkpoint, imgs_path=args.images, result_path=args.result)
