import tools.helpers as helpers
from tools.custom_datasets import name_to_dataset, make_dataset, CelebA
from robustness import train
import cox.store
from torchvision import datasets, transforms
import tools.transforms
import torch
import numpy as np
from torch.utils.data import Subset

if __name__ == "__main__":

    var_dict = helpers.get_runtime_inputs()

    helpers.set_seeds(var_dict)

    # do we transfer?
    is_Transfer = False
    if var_dict["source_eps"] > -1:
        is_Transfer = True

    # do we grab an imagenet pretrained model?
    pretrained = False
    if var_dict["source_eps"] == 0:
        pretrained = True

    # get dataset class
    dataset = make_dataset(var_dict)

    model = helpers.load_model(var_dict, is_Transfer, pretrained, dataset)

    model = helpers.change_linear_layer_out_features(
        model, var_dict, dataset, is_Transfer
    )

    if is_Transfer:
        model = helpers.re_init_and_freeze_blocks(model, var_dict)

    subset = var_dict["num_training_images"]
    if var_dict["num_training_images"] == -1:
        subset = None

    if var_dict["target_dataset_name"] == "celeba":
        train_set = CelebA(
            "./data",
            transform=tools.transforms.TRAIN_TRANSFORMS_DEFAULT(224),
        )
        test_set = CelebA(
            "./data",
            split="valid",
            transform=tools.transforms.TEST_TRANSFORMS_DEFAULT(224),
        )
        train_sample_count = len(train_set)
        rng = np.random.RandomState(var_dict["seed"])
        subset = rng.choice(list(range(train_sample_count)), size=subset, replace=False)
        # subset = subset[subset_start:]
        train_set = Subset(train_set, subset)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=var_dict["batch_size"],
            shuffle=True,
            num_workers=var_dict["num_workers"],
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=var_dict["batch_size"],
            shuffle=True,
            num_workers=var_dict["num_workers"],
            pin_memory=True,
        )

    else:
        train_loader, test_loader = dataset.make_loaders(
            workers=var_dict["num_workers"],
            batch_size=var_dict["batch_size"],
            subset=subset,
            subset_seed=var_dict["seed"],
        )

    out_store = helpers.make_out_store(var_dict)
    train_args = helpers.make_train_args(var_dict)

    helpers.print_details(model, var_dict, train_args)

    train.train_model(train_args, model, (train_loader, test_loader), store=out_store)
    pass
