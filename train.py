"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging

import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image 

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
Image.MAX_IMAGE_PIXELS = 1000000000 

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument("--data_root", type=str, default='/home/liqian/research/seg/dataset/mine_10x_6/train', 
                    help="path to Dataset")

parser.add_argument('--w', type=float, default=0.5, help='weight of MSCL loss')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepV3Plus')
parser.add_argument('--dataset', nargs='*', type=str, default=['S03'],
                    help='a list of datasets')

parser.add_argument('--val_dataset', nargs='*', type=str, default=['S01','S02','S03','S04','S05','S06'],
                    help='extra val_dataset')
parser.add_argument('--class_uniform_pct', type=float, default=0.5,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')
parser.add_argument('--img_wt_loss',  default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', default=False,
                    help='class-weighted loss')
parser.add_argument('--jointwtborder', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=10000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.5, help='level of color augmentation')
parser.add_argument('--gblur', default=True,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=32,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=224,
                    help='training crop size')

parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)


parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='6',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='Tissue-SDG_r50os16',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,1,1,1,0,0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--relax_denom', type=float, default=0.0)
parser.add_argument('--clusters', type=int, default=30)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')

parser.add_argument('--jit_only',  default=True)
# Contrastive arguments
parser.add_argument('--use_mscl', default=True)
parser.add_argument('--nce_T', type=float, default=0.07)
parser.add_argument('--contrast_temperature', type=float, default=0.1)
parser.add_argument('--contrast_max_classes', type=int, default=6)
parser.add_argument('--contrast_max_views', type=int, default=30)


args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

if args.test_mode:
    args.max_epoch = 2

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    train_loader, val_loaders, train_obj, extra_val_loaders = datasets.setup_loaders(args)  
    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args, criterion, criterion_aux)

    optim, scheduler = optimizer.get_optimizer(args, net)

    net.cuda()

    epoch = 0
    i = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        print("Continue Training")
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0
    else:
        print("New Training")

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter)
        
        for dataset, val_loader in val_loaders.items():
            ("Validation after epochs")
            validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)
            
        if args.local_rank == 0:
            print("Saving pth file...")
            evaluate_eval(args, net, optim, scheduler, None, None, [],
                        writer, epoch, "None", None, i, save_pth=True)

        epoch += 1

    for dataset, val_loader in val_loaders.items():
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i)


    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)


def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter, style_loader=None):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()

    torch.autograd.set_detect_anomaly(True)

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)
    
    for i, data in enumerate(train_loader): 
        if curr_iter >= max_iter:
            break

        inputs, inputs_color, gts, _,  aux_gts, index = data  

        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            if args.jit_only:
                inputs_color = [inputs_color]
            gts = [gts]
            aux_gts = [aux_gts]

        step = curr_iter / B
        # batch_pixel_size = C * H * W

        for di, ingredients in enumerate(zip(inputs, inputs_color, gts, aux_gts)):
            input, input_color, gt, aux_gt = ingredients  
            start_ts = time.time()

            img_gt = None
            input, input_color, gt, aux_gt = input.cuda(), input_color.cuda(), gt.cuda(), aux_gt.cuda()
            
            optim.zero_grad()

            outputs, logits, logits_j = net([input, input_color], gts=gt, aux_gts=aux_gt, img_gt=img_gt, global_step=step)
              
            outputs_index = 0
            main_loss = outputs[outputs_index]
            outputs_index += 1
            aux_loss = outputs[outputs_index]
            outputs_index += 1
            total_loss = main_loss + (0.4 * aux_loss)

            if args.use_mscl:
                _mscl = outputs[outputs_index]
                outputs_index += 1
                total_loss = total_loss + (args.w * _mscl)
          
            log_total_loss = total_loss.clone().detach_()
            probability_o = F.softmax(logits,dim=1)
            probability_j = F.softmax(logits_j,dim=1)
            probability = (probability_o + probability_j)/2
            entropy = -torch.sum(probability * torch.log(probability + 1e-8), dim=1)

            magnitude = torch.mean(entropy, dim=[1, 2])/np.log(6)
            train_loader.dataset.set_MAGNITUDE(index, 1 - magnitude.detach().cpu())  

            total_loss.backward()
            optim.step()
            time_meter.update(time.time() - start_ts)
 
            del total_loss, log_total_loss

        curr_iter += 1
        scheduler.step()

        if i > 5 and args.test_mode:
            return curr_iter
        if curr_epoch >= 10:
            train_loader.dataset.is_warmup_finished = True

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    dump_images = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes
        gt_cuda = gt_cuda.long()

        # output = F.log_softmax(output, dim=1)
        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    # torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    return val_loss.avg


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = 'feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()
