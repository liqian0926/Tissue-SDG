"""
Network Initializations
"""
import logging
import importlib
import datasets

def get_net(args, criterion, criterion_aux=None):
    """
    Get Network Architecture based on arguments provided
    """
    net = get_model(args=args, num_classes=datasets.num_classes,
                    criterion=criterion, criterion_aux=criterion_aux)
    num_params = sum([param.nelement() for param in net.parameters()]) 
    logging.info('Model params = {:2.3f}M'.format(num_params / 1000000)) 

    net = net.cuda()
    return net

def get_model(args, num_classes, criterion, criterion_aux=None):
    """
    Fetch Network Function Pointer
    """
    network = args.arch
    module = network[:network.rfind('.')] 
    model = network[network.rfind('.') + 1:] 
    mod = importlib.import_module(module) 
    net_func = getattr(mod, model) 
    net = net_func(args=args, num_classes=num_classes, criterion=criterion, criterion_aux=criterion_aux) 
    return net
