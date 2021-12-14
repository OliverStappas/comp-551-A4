# **Group 65 Reproducibility**

## **Packages Used**
- PyTorch
- [Robustness](https://github.com/MadryLab/robustness)

## **Pretrained Models**
Download all source models into models directory and save with the appropriate name:
- https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=0; save as 'imagenet_l2_3_0.pt'
- https://www.dropbox.com/s/axfuary2w1cnyrg/imagenet_linf_4.pt?dl=0; save as 'imagenet_linf_4.pt'
- https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0; save as 'imagenet_linf_8.pt'

## **Reproducibility**

For all experiments, once the model has finished running, change directory into /results

run `python ./mass_extractor.py` which produces a csv file with all the data

### **First/Second/Third Experiment**
From the root of the project run the command\
`python ./train.py -e 0 -t kmnist -ub 1 -b 128 -n 800 -w 2 -ne 100 -li 20 -s 20100000 -lr 0.1`

For a more detailed view of each of the parameters run: `python ./train.py -h`

Change parameters accordingly. See writeup for more details.

### **Fourth Experiment**

Download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

From the root of the project run the command\
`python ./train.py -e 0 -t celeba -ub 1 -b 128 -n 100 -w 2 -ne 100 -li 20 -s 20100000 -lr 0.1`

Ensure that the transforms applied in the train.py are as follows:\
`train_set = CelebA("./data",
transform=tools.transforms.TRAIN_TRANSFORMS_DEFAULT(224))`

`test_set = CelebA("./data",
            split="valid",
            transform=tools.transforms.TEST_TRANSFORMS_DEFAULT(224))`

### **Fifth Experiment**

Download the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

From the root of the project run the command\
`python ./train.py -e 0 -t celeba -ub 1 -b 128 -n 100 -w 2 -ne 100 -li 20 -s 20100000 -lr 0.1`

Ensure that the transforms applied in the train.py are as follows:\
`train_set = CelebA("./data",
transform=tools.transforms.TRAIN_TRANSFORMS_DOWNSCALE(32,224))`

`test_set = CelebA("./data",
            split="valid",
            transform=tools.transforms.TEST_TRANSFORMS_DOWNSCALE(32,224))`
            
            
### **Sixth  Experiment**

This requires having existing models trained on eps 3 and 0 for 100 images and 3 blocks on cifar10\
In hessian.py and in make_grads.py, replace 3: ... and 0: ... with your path to those models\

Then, from the inlfuence_functions path run the commands\
` python ./hessian.py -e 0 -ds cifar10`\
` python ./hessian.py -e 3 -ds cifar10`

Then,\
` python ./make_grads.py -e 0 -ds cifar10`\
` python ./make_grads.py -e 3 -ds cifar10`

Then,\
` python ./make_influences.py -e 0 -ds cifar10`\
` python ./make_influences.py -e 3 -ds cifar10`

You may then proceed with running the visualizing influence functions.ipynb notebook to get the influence function results
