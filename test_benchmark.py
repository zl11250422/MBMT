# Copyright (c) 2022 Yawei Li, Kai Zhang, Radu Timofte SPDX license identifier: MIT
# This file may have been modified by ByteDance
import os.path
import logging
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchstat import stat
from utils import utils_logger
from utils import utils_image as util
from utils.model_summary import get_model_flops, get_model_activation
#from model.rlfn_ntire import RLFN_Prune
#from model.rlfn import RLFN
from basicsr.archs.dat_arch import DAT
from calflops import calculate_flops

def main():

    utils_logger.logger_info('NTIRE2022-EfficientSR', log_path='NTIRE2022-EfficientSR.log')
    logger = logging.getLogger('NTIRE2022-EfficientSR')

    # --------------------------------
    # basic settings
    # --------------------------------
    # testsets = 'DIV2K 901-1000'
    testsets = os.path.join(os.getcwd(), 'data')
    testset_L = 'DIV2K_test_LR'

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # load model
    # --------------------------------
    #model_path = os.path.join('model_zoo', 'rlfn_x4.pth')
    model_path = r'H:\down\SISR\mbmt\experiments\ablation_2branches_sa_dw\models\net_g_474000.pth'
    model = DAT(upscale=4,in_chans=3,img_size=64,img_range=1.,depth=[8],embed_dim=60,num_heads=[6],expansion_factor=2,resi_connection='3conv',split_size=[8,32],upsampler='pixelshuffledirect')
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    flops, macs, params = calculate_flops(model=model, input_shape=(1,3,320,180)) # default input shape: (1, 128)
    print("FLOPs:%s  MACs:%s  Params:%s \n" %(flops, macs, params))
    #stat(model, (3, 224, 224))
    exit()
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    #L_folder = os.path.join(testsets, testset_L)
    L_folder = r'C:\Users\lzhan\Desktop\sisr\SISR\BSRN\BSRN\datasets\Set14\LRbicx4'
    
    E_folder = os.path.join(testsets, testset_L+'_results')
    util.mkdir(E_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L * 255.
        img_L = img_L.to('cpu')

        start.record()
        img_E = F.interpolate(img_L, scale_factor=4, mode="bicubic")

        #img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = img_E / 255.
        img_E = util.tensor2uint(img_E)

        util.imsave(img_E, os.path.join(E_folder, img_name[:]+ext))

    input_dim = (3, 320, 180)  # set the input dimension
    #stat(model.to(torch.device('cpu')), input_dim)
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

if __name__ == '__main__':

    main()
