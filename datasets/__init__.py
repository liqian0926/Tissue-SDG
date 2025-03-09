"""
Dataset setup and loaders
"""

from utils import ext_transforms as et
from torch.utils.data import DataLoader
from datasets.ours import OurSegmentation, TestSegmentation

num_classes = 6
ignore_label = 255

def create_extra_val_loader(args, dataset, val_transform, val_sampler):
    """
    Create extra validation loader
    Args:
        args: input config arguments
        dataset: dataset class object
        val_input_transform: validation input transforms
        target_transform: target transforms
        val_sampler: validation sampler

    return: validation loaders
    """

    val_set = TestSegmentation(root='/home/liqian/research/seg/dataset/mine_10x_6/test/'+dataset,
                              image_set='test',transform=val_transform)
    val_sampler = None
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                            sampler = val_sampler)
    return val_loader


def setup_loaders(args):
    """
    Setup Data Loaders[Currently supports Cityscapes, Mapillary and ADE20kin]
    input: argument passed by the user
    return:  training data loader, validation data loader loader,  train_set
    """

    args.train_batch_size = args.bs_mult
    if args.bs_mult_val > 0:
        args.val_batch_size = args.bs_mult_val
    else:
        args.val_batch_size = args.bs_mult

    args.num_workers = 8 
    if args.test_mode:
        args.num_workers = 1
    val_loaders = {}

    
    train_transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(224,224), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_set = OurSegmentation(root=args.data_root,
                                image_set='train', transform=train_transform)

    val_set = OurSegmentation(root=args.data_root,
                              image_set='val',transform=val_transform)
        

    val_sampler = None
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size,
                            num_workers=args.num_workers // 2 , shuffle=False, drop_last=False,
                            sampler = val_sampler)

    val_loaders['S03'] = val_loader
    train_sampler = None
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size,
                              num_workers=args.num_workers, shuffle=(train_sampler is None), drop_last=True, sampler = train_sampler)
    extra_val_loader = {}
    for val_dataset in args.val_dataset:
        extra_val_loader[val_dataset] = create_extra_val_loader(args, val_dataset, val_transform, val_sampler)

    return train_loader, val_loaders, train_set, extra_val_loader
