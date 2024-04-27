# test.py
import os
import torch
import argparse
from torchvision import transforms
from PIL import Image
from model import PhysicalNN

def main(checkpoint, imgs_path, result_path):
    ori_dirs = []
    for image in os.listdir(imgs_path):
        ori_dirs.append(os.path.join(imgs_path, image))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)

    print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    model.eval()

    testtransform = transforms.Compose([
        transforms.ToTensor(),
    ])
    unloader = transforms.ToPILImage()

    for imgdir in ori_dirs:
        img_name = os.path.splitext(os.path.basename(imgdir))[0]
        img = Image.open(imgdir)
        inp = testtransform(img).unsqueeze(0)
        inp = inp.to(device)
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        dir = os.path.join(result_path, 'results_{}'.format(checkpoint['epoch']))
        if not os.path.exists(dir):
            os.makedirs(dir)
        corrected.save(os.path.join(dir, '{}_corrected.png'.format(img_name)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='checkpoints path', required=True)
    parser.add_argument('--images', help='test images folder', default='./test_img/')
    parser.add_argument('--result', help='results folder', default='./results/')
    args = parser.parse_args()
    checkpoint = args.checkpoint
    imgs = args.images
    result_path = args.result
    main(checkpoint=checkpoint, imgs_path=imgs, result_path=result_path)
