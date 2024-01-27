import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--index', type=int, default="25",
                    help='the index for leaking images on Dataset.')
parser.add_argument('--image', type=str, default="",
                    help='the path to customized image.')
parser.add_argument('--defense', type=bool, default=True, help='Defend the deep gradient leakage or not.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

if args.dataset == 'cifar10':
    dst = datasets.CIFAR10("../../data/cifar10", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device)

if len(args.image) > 1:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device)


gt_data = gt_data.view(1, *gt_data.size())
gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tt(gt_data[0].cpu()))
if args.defense:
    plt.savefig('./leakage_image_defense/origin_data.png')
    print('Defend Attack!')
else:
    plt.savefig('./leakage_image/origin_data.png')

from models.vision import LeNet, weights_init
net = LeNet().to(device)


torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot


pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = (list(_.detach().clone() for _ in dy_dx))

def gaussian_noise(data_shape, sigma, device=None):
    """
    Gaussian noise
    """
    return np.random.normal(0, sigma, data_shape)

if args.defense:
    original_dy_dx = original_dy_dx + gaussian_noise(original_dy_dx.shape, 1, device)

dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))
if args.defense:
    plt.savefig('./leakage_image_defense/init_dummy_data.png')
else:
    plt.savefig('./leakage_image/init_dummy_data.png')

optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=0.01)

history = []
for iters in range(3000):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % 100 == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

plt.figure(figsize=(12, 8))
for i in range(30):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    if args.defense:
        plt.savefig('./leakage_image_defense/dummy_image_{}.png'.format(i * 10))
    else:
        plt.savefig('./leakage_image/dummy_image_{}.png'.format(i * 10))
    plt.title("iter=%d" % (i * 10))
    plt.axis('off')

plt.show()
