import torchvision.transforms as standard_transforms
import transforms.transforms as extended_transforms
import torchvision.transforms as transforms

from augmentation.DyADA import dyada
from augmentation import trivialaugment


def magnitude_dyada(magnitude):

    trivialaugment.set_augmentation_space(augmentation_space='standard', magnitude=magnitude)
    magnitude_transform = transforms.Compose([
                standard_transforms.Compose([standard_transforms.RandomApply([
                standard_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
                extended_transforms.RandomBilateralBlur(),
                extended_transforms.RandomGaussianBlur(),
                standard_transforms.ToTensor()
                ]),
                transforms.ToPILImage(),
                dyada(M=magnitude),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
            ])
    return magnitude_transform

def make_transform(magnitude):

    trivialaugment.set_augmentation_space(augmentation_space='standard',magnitude=int(magnitude*10))
    transform = transforms.Compose([
                transforms.ToPILImage(),
                trivialaugment.TrivialAugment(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
                ])
    return transform, transform_test