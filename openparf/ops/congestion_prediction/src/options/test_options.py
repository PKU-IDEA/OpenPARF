import argparse
import os
from ..util import util
import torch

class TestOptions():
    def __init__(self):
        self.subparser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        #test options
        self.subparser.add_argument('--config', type=str, help='benchmark address')
        self.subparser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.subparser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.subparser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.subparser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.subparser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.subparser.add_argument('--how_many', type=int, default=300, help='how many test images to run')
        self.subparser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.subparser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.subparser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.subparser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.subparser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
        self.isTrain = False

        # experiment specifics
        self.subparser.add_argument('--name', type=str, default='rudy_congestion', help='name of the experiment. It decides where to store samples and models')
        self.subparser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.subparser.add_argument('--checkpoints_dir', type=str, default='./openparf/ops/congestion_prediction/checkpoints', help='models are saved here')
        self.subparser.add_argument('--model', type=str, default='pix2pixHD', help='which model to use')
        self.subparser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.subparser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.subparser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], help="Supported data type i.e. 8, 16, 32 bit")
        self.subparser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # input/output sizes
        self.subparser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.subparser.add_argument('--loadSize', type=int, default=480, help='scale images to this size')
        self.subparser.add_argument('--fineSize', type=int, default=480, help='then crop to this size')
        self.subparser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.subparser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.subparser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.subparser.add_argument('--dataroot', type=str, default='/home/abc/xyc/datasets/')
        self.subparser.add_argument('--resize_or_crop', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.subparser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.subparser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        self.subparser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.subparser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.subparser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.subparser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.subparser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.subparser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer') # this value was originally 64
        self.subparser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        self.subparser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.subparser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.subparser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        self.subparser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        self.subparser.add_argument('--no_instance', action='store_false', help='if specified, do *not* add instance map as input')
        self.subparser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.subparser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')
        self.subparser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.subparser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.subparser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.subparser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.subparser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.subparser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt
