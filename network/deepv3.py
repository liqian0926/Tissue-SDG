
import torch
from torch import nn
from network import Resnet
from network.mynn import initialize_weights, Norm2d, Upsample,  unfreeze_weights
from network.mscl import Class_PixelNCELoss, ProjectionHead

import torchvision.models as models


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x_o depth
      3x3 x_o depth dilation 6
      3x3 x_o depth dilation 12
      3x3 x_o depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x_o):
        x_size = x_o.size()

        img_features = self.img_pooling(x_o)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x_o)
            out = torch.cat((out, y), 1)
        return out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLab-V3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='resnet-50', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.trunk = trunk

        if self.training:
            self.criterion_CA = nn.MSELoss()
            self.criterion_MSCL = Class_PixelNCELoss(self.args)


        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-18':
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 64
            channel_4th = 128
            prev_final_channel = 256
            final_channel = 512
            resnet = Resnet.resnet18(wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-50':
            resnet = Resnet.resnet50(wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101': # three 3 X 3
            resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
                                        resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        elif trunk == 'resnet-152':
            resnet = Resnet.resnet152()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-50':
            resnet = models.resnext50_32x4d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101':
            resnet = models.resnext101_32x8d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-50':
            resnet = models.wide_resnet50_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-101':
            resnet = models.wide_resnet101_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D4':
            for n, m in self.layer2.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (8, 8), (8, 8), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
        else:
            os = 32

        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                    output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        
        initialize_weights(self.dsn)
        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final1)
        initialize_weights(self.final2) 

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if trunk == 'resnet-101':
            self.three_input_layer = True
            in_channel_list = [64, 64, 128, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [32, 32, 64, 128, 256,  512, 1024]
        elif trunk == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 32, 64,  128, 256]
        elif trunk == 'shufflenetv2':
            self.three_input_layer = False
            in_channel_list = [0, 0, 24, 116, 232, 464, 1024]
        elif trunk == 'mobilenetv2':
            self.three_input_layer = False
            in_channel_list = [0, 0, 16, 32, 64, 320, 1280]
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]

        # Projection head for Contrastive Learning
        self.ProjectionHead_cls_0 = ProjectionHead(dim_in=in_channel_list[-1])
        self.ProjectionHead_cls_1 = ProjectionHead(dim_in=304)
        self.ProjectionHead_cls_2 = ProjectionHead(dim_in=256)

        initialize_weights(self.ProjectionHead_cls_0)
        initialize_weights(self.ProjectionHead_cls_1)
        initialize_weights(self.ProjectionHead_cls_2)


    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, apply_bound=False, global_step=None):

        k_arr = []
        q_arr = []

        kd_arr = []
        qd_arr = []

        if len(x) == 2:
            x_o = x[0]
            x_j = x[1]
        else:
            x_o = x

        x_size = x_o.size()  

        x_o = self.layer0[0](x_o)
        if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
            x_o, k = self.layer0[1](x_o)
            k_arr.append(k)
        else:
            x_o = self.layer0[1](x_o)
        x_o = self.layer0[2](x_o)
        x_o = self.layer0[3](x_o)

        if self.training and self.args.jit_only:
            x_j = self.layer0[0](x_j)
            x_j, q = self.layer0[1](x_j)
            q_arr.append(q)
            x_j = self.layer0[2](x_j)
            x_j = self.layer0[3](x_j)

        x_tuple = self.layer1([x_o, k_arr])  
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  
        x_tuple = self.layer3(x_tuple)  
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  
        x_o = x_tuple[0]
        k_arr = x_tuple[1]
        kd_arr.append(x_o)
        x_o = self.aspp(x_o)
        dec0_up = self.bot_aspp(x_o)

        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        kd_arr.append(dec0)
        dec1 = self.final1(dec0)
        kd_arr.append(dec1)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])
        main_out_out = main_out.clone()
        if self.training:
            if self.args.jit_only:
                x_j_tuple = self.layer1([x_j, q_arr])  
                low_level_j = x_j_tuple[0]

                x_j_tuple = self.layer2(x_j_tuple)  
                x_j_tuple = self.layer3(x_j_tuple)  

                x_j_tuple = self.layer4(x_j_tuple)  
                x_j = x_j_tuple[0]
                q_arr = x_j_tuple[1]

                qd_arr.append(x_j)
                x_j = self.aspp(x_j)
                dec0_up_j = self.bot_aspp(x_j)
                dec0_fine_j = self.bot_fine(low_level_j)
                dec0_up_j = Upsample(dec0_up_j, low_level_j.size()[2:])
                dec0_j = [dec0_fine_j, dec0_up_j]
                dec0_j = torch.cat(dec0_j, 1)
                qd_arr.append(dec0_j)
                dec1_j = self.final1(dec0_j)
                qd_arr.append(dec1_j)
                dec2_j = self.final2(dec1_j)
                main_out_j = Upsample(dec2_j, x_size[2:])
                main_out_j_out = main_out_j.clone()

            gts = gts.long()
            loss1 = (self.criterion(main_out, gts) + self.criterion(main_out_j, gts)) / 2



            aux_out = self.dsn(aux_out)
            if aux_gts.dim() == 1:
                aux_gts = gts
            aux_gts = aux_gts.unsqueeze(1).float()
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            aux_gts = aux_gts.squeeze(1).long()
            loss2 = self.criterion_aux(aux_out, aux_gts)

            return_loss = [loss1, loss2]
            
            if self.args.jit_only:
                _, predict = torch.max(dec2, 1)
                _, predict_j = torch.max(dec2_j, 1)

                if self.args.use_mscl:
                    mscl = torch.FloatTensor([0]).cuda()

                if self.args.use_mscl:
                    for N, f_maps in enumerate(zip(qd_arr, kd_arr)):
                        feat_q, feat_k = f_maps

                        projection = getattr(self, 'ProjectionHead_cls_%d' % N)

                        embed_q = projection(feat_q)
                        embed_k = projection(feat_k)

                        if self.args.use_mscl:
                            loss_cw = self.criterion_MSCL(embed_q, embed_k, gts, predict)
                            mscl = mscl + loss_cw.mean()

                if self.args.use_mscl:
                    mscl = mscl / len(qd_arr)
                    return_loss.append(mscl)

            return return_loss, main_out_out, main_out_j_out
        else:
            return main_out

def get_final_layer(model):
    unfreeze_weights(model.final)
    return model.final


def DeepR18V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 18 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-18")
    return DeepV3Plus(num_classes, trunk='resnet-18', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepR50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepR50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-50")
    return DeepV3Plus(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-101")
    return DeepV3Plus(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepR152V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 152 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNet-152")
    return DeepV3Plus(num_classes, trunk='resnet-152', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)



def DeepResNext50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-50 32x4d")
    return DeepV3Plus(num_classes, trunk='resnext-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepResNext101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Resnext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : ResNext-101 32x8d")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-50")
    return DeepV3Plus(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepWideResNet101V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : wide_resnet-101")
    return DeepV3Plus(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepResNext101V3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepResNext101V3PlusD_OS4(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3+, Backbone : resnext-101")
    return DeepV3Plus(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D4', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS32(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepMNASNet05V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_0_5")
    return DeepV3Plus(num_classes, trunk='mnasnet_05', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMNASNet10V3PlusD(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3+, Backbone : mnas_1_0")
    return DeepV3Plus(num_classes, trunk='mnasnet_10', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)


def DeepShuffleNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : mobilenetv2")
    return DeepV3Plus(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepShuffleNetV3PlusD_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3+, Backbone : shufflenetv2")
    return DeepV3Plus(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
