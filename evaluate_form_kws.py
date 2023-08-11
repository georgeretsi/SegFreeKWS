import argparse
import logging

import os
import numpy as np
import torch.cuda
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.iam_dataset import IAMDataset

#from eval_config import *


from dataclasses import dataclass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('HTR-Experiment::train')
logger.info('--- Running HTR Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--dataset_path', action='store', type=str, default='../../datasets/')
parser.add_argument('--model_path',  action='store', type=str, default='./saved_models/temp_best.pt')
parser.add_argument('--dataset',  action='store', type=str, default='iam')
parser.add_argument('--clevels',  action='store', type=int, default=1)
parser.add_argument('--cos_thres',  action='store', type=float, default=0.5)
parser.add_argument('--ctc_thres',  action='store', type=float, default=2.5)
parser.add_argument('--K',  action='store', type=int, default=30)
parser.add_argument('--prob_thres',  action='store', type=float, default=0.05)
parser.add_argument('--carea_ratio',  action='store', type=float, default=0.5)
parser.add_argument('--ctc_mode',  action='store', type=int, default=2, help='0: no ctc, 1: only forward ctc, 2: forward and backward ctc')
parser.add_argument('--iou_mode',  action='store', type=int, default=2, help='0: typical iou, 1: x-iou, 2: iow')
parser.add_argument('--doc_scale',  action='store', type=float, default=1.0)

args = parser.parse_args()


gpu_id = args.gpu_id

logger.info('Loading dataset.')

dataset = args.dataset
dataset_folder = args.dataset_path
if dataset != 'iam':
    raise NotImplementedError

# masked form to remove erroneous lines of IAM
masked_form = (dataset == 'iam')
# KWS typically evaluated on a reduced character set without punctuations etc.
# reduced character set to be in line with KWS methods
reduced_charset = True

form_test_set = IAMDataset(dataset_folder, subset='test', segmentation_level='form', fixed_size=None, transforms=None)

# character set - to be pruned if reduced character set is true
classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '


# load CNN
logger.info('Preparing Net...')

cnn = torch.load(args.model_path)
cnn.cuda(args.gpu_id)
cnn.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(' CNN #parameters : ' + str(count_parameters(cnn)))


logger.info('Evaluating:')

from evaluation_functions import seg_free_eval

@dataclass
class EvalArgs:
    clevels: int = args.clevels
    cos_thres: float = args.cos_thres
    ctc_thres: float = args.ctc_thres
    K: int = args.K
    prob_thres: float = args.prob_thres
    carea_ratio: float = args.carea_ratio
    ctc_mode: int = args.ctc_mode
    iou_mode: int = args.iou_mode
    doc_scale: float = args.doc_scale
    masked_form: bool = masked_form
    reduced_charset: bool = reduced_charset

#eval_args = EvalArgs(ctc_thres=-1)
eval_args = EvalArgs()

map_metrics = seg_free_eval(form_test_set, cnn, classes, eval_args, eval_multiple_thres=True)