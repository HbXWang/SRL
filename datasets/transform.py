import cv2
import numpy as np
import torchvision


class GaussianBlur:
    def __init__(self, kernel_size, x_min=0.1, x_max=2.0):
        self.min = x_min
        self.max = x_max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


class SRL_Transforms:
    def __init__(self, size, s=1.0, mean=None, std=None, blur=False, k_crops=2):
        self.train_transform = [
            torchvision.transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)],
                                               p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
        ]
        if blur:
            self.train_transform.append(GaussianBlur(kernel_size=int(0.1 * size)))
        self.train_transform.append(torchvision.transforms.ToTensor())
        self.test_transform = [
            torchvision.transforms.Resize(size=(size, size)),
            torchvision.transforms.ToTensor(),
        ]
        if mean and std:
            self.train_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
            self.test_transform.append(torchvision.transforms.Normalize(mean=mean, std=std))
        self.train_transform = torchvision.transforms.Compose(self.train_transform)
        self.test_transform = torchvision.transforms.Compose(self.test_transform)

        self.k_crops = k_crops

    def __call__(self, x):
        total_out = [self.train_transform(x)]
        for j in range(self.k_crops):
            k = self.train_transform(x)
            total_out.append(k)
        return total_out
