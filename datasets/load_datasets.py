from torch.utils import data
from torchvision import datasets


def load_datasets(dataset_name, dataset_dir, x_transform):
    if dataset_name == "stl-10":
        train_dataset = datasets.STL10(root=dataset_dir, download=True, split="train", transform=x_transform)
        test_dataset = datasets.STL10(root=dataset_dir, download=True, split="test", transform=x_transform)
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif dataset_name == "cifar-10":
        train_dataset = datasets.CIFAR10(root=dataset_dir, download=True, train=True, transform=x_transform)
        test_dataset = datasets.CIFAR10(root=dataset_dir, download=True, train=False, transform=x_transform)
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif dataset_name == "cifar-100":
        train_dataset = datasets.CIFAR100(root=dataset_dir, download=True, train=True, transform=x_transform)
        test_dataset = datasets.CIFAR100(root=dataset_dir, download=True, train=False, transform=x_transform)
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 20
    elif dataset_name == "ImageNet-10":
        dataset = datasets.ImageFolder(root=dataset_dir, transform=x_transform)
        class_num = 10
    elif dataset_name == "ImageNet-dogs":
        dataset = datasets.ImageFolder(root=dataset_dir, transform=x_transform)
        class_num = 15
    else:
        raise NotImplementedError
    return dataset, class_num


def load_datasets_draw(dataset_name, dataset_dir, x_transform):
    if dataset_name == "stl-10":
        dataset = datasets.STL10(root=dataset_dir, download=True, split="test", transform=x_transform)
        class_num = 10
    elif dataset_name == "cifar-10":
        dataset = datasets.CIFAR10(root=dataset_dir, download=True, train=False, transform=x_transform)
        class_num = 10
    elif dataset_name == "cifar-100":
        dataset = datasets.CIFAR100(root=dataset_dir, download=True, train=False, transform=x_transform)
        class_num = 20
    else:
        raise NotImplementedError
    return dataset, class_num
