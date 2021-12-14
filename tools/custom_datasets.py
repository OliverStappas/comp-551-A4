import tools.transforms as transforms
import tools.constants as constants
import os
from robustness import imagenet_models, cifar_models
from robustness.datasets import DataSet, CIFAR
import torch as ch
from torchvision import datasets
from functools import partial
import torch
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets.vision import VisionDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetTransfer(DataSet):
    def __init__(self, data_path, num_transf_classes=1000, **kwargs):
        self.num_classes = num_transf_classes
        imagenet_size = 224
        transform_type_to_transform = {
            "default": (
                transforms.TRAIN_TRANSFORMS_DEFAULT(imagenet_size),
                transforms.TEST_TRANSFORMS_DEFAULT(imagenet_size),
            ),
            "black_n_white": (
                transforms.BLACK_N_WHITE(imagenet_size),
                transforms.BLACK_N_WHITE(imagenet_size),
            ),
        }
        if kwargs["downscale"]:
            transform_type_to_transform = {
                "default": (
                    transforms.TRAIN_TRANSFORMS_DOWNSCALE(
                        kwargs["downscale_size"], imagenet_size
                    ),
                    transforms.TEST_TRANSFORMS_DOWNSCALE(
                        kwargs["downscale_size"], imagenet_size
                    ),
                ),
                "black_n_white": (
                    transforms.BLACK_N_WHITE_DOWNSCALE(
                        kwargs["downscale_size"], imagenet_size
                    ),
                    transforms.BLACK_N_WHITE_DOWNSCALE(
                        kwargs["downscale_size"], imagenet_size
                    ),
                ),
            }
        ds_kwargs = {
            "num_classes": kwargs["num_classes"],
            "mean": ch.tensor(kwargs["mean"]),
            "std": ch.tensor(kwargs["std"]),
            "custom_class": kwargs["custom_class"],
            "label_mapping": None,
            "transform_train": transform_type_to_transform[kwargs["transform_type"]][0],
            "transform_test": transform_type_to_transform[kwargs["transform_type"]][1],
        }
        #        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        self.name = kwargs["name"]
        super(ImageNetTransfer, self).__init__(kwargs["name"], data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        return imagenet_models.__dict__[arch](num_classes=1000, pretrained=pretrained)


CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

# this class is used when we're training from scratch instead of from a pre-trained imagenet model
class CIFAR(DataSet):
    def __init__(self, num_classes, data_path=None, **kwargs):
        self.name = f"cifar{num_classes}"

        num_classes_to_custom_class = {10: datasets.CIFAR10, 100: datasets.CIFAR100}

        ds_kwargs = {
            "num_classes": num_classes,
            "mean": ch.tensor(CIFAR_MEAN),
            "std": ch.tensor(CIFAR_STD),
            "custom_class": num_classes_to_custom_class[num_classes],
            "label_mapping": None,
            "transform_train": transforms.TRAIN_TRANSFORMS_DEFAULT(32),
            "transform_test": transforms.TEST_TRANSFORMS_DEFAULT(32),
        }
        super(CIFAR, self).__init__(f"cifar{num_classes}", data_path, **ds_kwargs)

    def get_model(self, arch, pretrained=False):
        if pretrained:
            raise ValueError("CIFAR100 does not support pytorch_pretrained=True")
        return cifar_models.__dict__[arch](num_classes=num_classes)


class CelebA(VisionDataset):

    base_folder = "celeba"
    file_list = [
        # File ID                         MD5 Hash                            Filename
        (
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
            "00d2c5bc6d35e252742224ab0c1e8fcb",
            "img_align_celeba.zip",
        ),
        (
            "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "75e246fa4810816ffd6ee81facbd244c",
            "list_attr_celeba.txt",
        ),
        (
            "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
            "32bd1bd63d3c78cd57e08160ec5ed1e2",
            "identity_CelebA.txt",
        ),
        (
            "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
            "00566efa6fedff7a56946cd1c10f1c16",
            "list_bbox_celeba.txt",
        ),
        (
            "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
            "cc24ecafdb5b50baae59b03474781f8c",
            "list_landmarks_align_celeba.txt",
        ),
        (
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        index_attr=20,
    ) -> None:
        import pandas

        super(CelebA, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.index_attr = index_attr
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(
            fn("list_eval_partition.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
        )
        identity = pandas.read_csv(
            fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0
        )
        bbox = pandas.read_csv(
            fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0
        )
        landmarks_align = pandas.read_csv(
            fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1
        )
        attr = pandas.read_csv(
            fn("list_attr_celeba.txt"), delim_whitespace=True, header=1
        )

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "img_align_celeba", self.filename[index]
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, self.index_attr])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError('Target type "{}" is not recognized.'.format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)


name_to_dataset = {
    "caltech101_stylized": {
        "num_classes": 101,
        "custom_class": None,
        "transform_type": "default",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "food_stylized": {
        "num_classes": 101,
        "custom_class": None,
        "transform_type": "default",
        "mean": [0.5493, 0.4450, 0.3435],
        "std": [0.2730, 0.2759, 0.2800],
    },
    "caltech101": {
        "num_classes": 101,
        "custom_class": None,
        "transform_type": "default",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "food": {
        "num_classes": 101,
        "custom_class": None,
        "transform_type": "default",
        "mean": [0.5493, 0.4450, 0.3435],
        "std": [0.2730, 0.2759, 0.2800],
    },
    "cifar10": {
        "num_classes": 10,
        "custom_class": datasets.CIFAR10,
        "transform_type": "default",
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    "cifar100": {
        "num_classes": 100,
        "custom_class": datasets.CIFAR100,
        "transform_type": "default",
        "mean": CIFAR_MEAN,
        "std": CIFAR_STD,
    },
    "svhn": {
        "num_classes": 10,
        "custom_class": datasets.SVHN,
        "transform_type": "default",
        "mean": [0.4377, 0.4438, 0.4728],
        "std": [0.1980, 0.2010, 0.1970],
    },
    # TODO: Get mean and std for fmnist
    "fmnist": {
        "num_classes": 10,
        "custom_class": datasets.FashionMNIST,
        "transform_type": "black_n_white",
        "mean": [0.1801, 0.1801, 0.1801],
        "std": [0.3421, 0.3421, 0.3421],
    },
    "kmnist": {
        "num_classes": 10,
        "custom_class": datasets.KMNIST,
        "transform_type": "black_n_white",
        "mean": [0.1801, 0.1801, 0.1801],
        "std": [0.3421, 0.3421, 0.3421],
    },
    "mnist": {
        "num_classes": 10,
        "custom_class": datasets.MNIST,
        "transform_type": "black_n_white",
        "mean": [0.1307, 0.1307, 0.1307],
        "std": [0.3081, 0.3081, 0.3081],
    },
    "celeba": {
        "num_classes": 2,
        "custom_class": datasets.CelebA,
        "transform_type": "default",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
}


def make_dataset(args):
    return ImageNetTransfer(
        name=args["target_dataset_name"],
        data_path=f'{constants.base_data_path}{args["target_dataset_name"]}',
        downscale=args["downscale"],
        downscale_size=args["downscale_size"],
        **name_to_dataset[args["target_dataset_name"]],
    )
