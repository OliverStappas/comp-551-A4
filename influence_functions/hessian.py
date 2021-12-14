from robustness import tools, train
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet, CIFAR, CINIC
from torch import autograd
from torchvision import datasets
import torch as ch
import numpy as np
import argparse
import random
from scipy import linalg
import os
# note: currently only supported for CIFAR10 and epsilon=[0,3]
import sys
sys.path.insert(1, '..')
from tools.transforms import TEST_TRANSFORMS_DEFAULT
from tools.helpers import flatten_grad, make_mask, set_seeds, eval_hessian 
from tools.helpers import get_runtime_inputs_for_influence_functions
import tools.custom_datasets as custom_datasets

if __name__ == '__main__':
    args = get_runtime_inputs_for_influence_functions()
    eps        = args.e
    num_images = args.n
    batch_size = args.b
    seed       = args.s
    ub         = args.ub
    ds         = args.ds

    # LOAD MODEL
    eps_to_model = {3: '../results/logs/source_eps_3.0_target_dataset_cifar10_num_training_images_100_unfrozen_blocks_3_seed_20100000_downscaled_False_/39af574d-416c-4a83-a709-b47bfac28151/99_checkpoint.pt',
                    0: '../results/logs/source_eps_0.0_target_dataset_cifar10_num_training_images_100_unfrozen_blocks_3_seed_20100000_downscaled_False_/05235740-8072-4a3a-820f-f0baa4d55dee/99_checkpoint.pt'
                    }
    # {3: '../results/logs/source_eps_3.0_target_dataset_cifar10_num_training_images_3200_unfrozen_blocks_3_seed_20100000_downscaled_False_/270a87ee-9237-4c98-a5cc-48292116fef5/99_checkpoint.pt',
    #                 0: '../results/logs/source_eps_0.0_target_dataset_cifar10_num_training_images_3200_unfrozen_blocks_3_seed_20100000_downscaled_False_/606be8cb-fc35-4816-a283-2d083228974a/99_checkpoint.pt'
    #                 }
                    # {3: f'l2_{eps}_imagenet_to_{ds}_{ub}_ub_{num_images}_images.pt',
                    # 0: f'nat_imagenet_to_{ds}_{ub}_ub_{num_images}_images.pt'} 

    # source_model_path = '../models/' + eps_to_model[eps]
    source_model_path = eps_to_model[eps]

    num_images = 100


    model, _ = make_and_restore_model(arch='resnet50', dataset=ImageNet('/tmp', num_classes=10), 
                                    resume_path=source_model_path, parallel=False)
    model.eval()
    criterion = ch.nn.CrossEntropyLoss()

    # MAKE DATASET
    size = (224, 224)
    if ds == 'cifar10':
        data_set = datasets.CIFAR10(root='/tmp', train=True, download=True,
                                    transform=TEST_TRANSFORMS_DEFAULT(size))
    elif ds == 'svhn':
        data_set = datasets.SVHN(root='/tmp', split='train', download=True,
                                transform=TEST_TRANSFORMS_DEFAULT(size))
    # SET THE SEEDS
    set_seeds({'seed':1000000})

    # MAKE THE MASK
    dataset_size = len(data_set)
    mask_sampler = make_mask(num_images, data_set)

    # MAKE THE LOADER
    num_workers  = 2 # 10
    loader       = ch.utils.data.DataLoader(data_set, sampler=mask_sampler, batch_size=batch_size, 
                                    shuffle=False, num_workers=num_workers, pin_memory=True)

    print(len(loader))

    # MAKE THE HESSIAN
    for i, data in enumerate(loader):
        print((i+1)*100)
        image, label = data
        output, final_inp = model(image.cuda())
        output = output.cpu()
        loss = criterion(output, label).cpu().double()
        loss_grad = autograd.grad(loss, model.model.fc.parameters(), create_graph=True)
        loss_grad = (loss_grad[0].cpu().double(), loss_grad[1].cpu().double())
        H_i = eval_hessian(loss_grad, model.model.fc)
        if i > 0:
            H += H_i
        else:
            H = H_i
        ch.cuda.empty_cache()

    if 'hessians' not in os.listdir(): os.mkdir('hessians')
    np.save(f'hessians/H_{ds}_target_{ub}_ub_{eps}_eps_{num_images}_images_{batch_size}_batch_size', H)

    # INVERT THE HESSIAN
    H_inv = linalg.pinv2(H, rcond=5e-2)
    if 'h_inverses' not in os.listdir(): os.mkdir('h_inverses')
    np.save(f'h_inverses/H_inv_{ds}_target_{ub}_ub_{eps}_eps_{num_images}_images_{batch_size}_batch_size', H_inv)

