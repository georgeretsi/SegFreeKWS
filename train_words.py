import argparse
import logging

import numpy as np

import  torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import CNN
from utils.iam_dataset import IAMDataset

from utils.auxilary_functions import affine_transformation

from evaluation_functions import seg_free_eval

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
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                    help='lr')
parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
parser.add_argument('--dataset_path', action='store', type=str, default='../../datasets/')
parser.add_argument('--model_load_path',  action='store', type=str, default=None)
parser.add_argument('--model_save_path',  action='store', type=str, default='./saved_models/temp.pt')
parser.add_argument('--dataset',  action='store', type=str, default='iam')
parser.add_argument('--max_epochs',  action='store', type=int, default=80)
parser.add_argument('--batch_size',  action='store', type=int, default=64)

args = parser.parse_args()

gpu_id = args.gpu_id

logger.info('Loading dataset.')

max_epochs = args.max_epochs
batch_size = args.batch_size

# dataset loaders for training and testing

dataset = args.dataset
dataset_folder = args.dataset_path
if dataset != 'iam':
    raise NotImplementedError

aug_transforms =[lambda x: affine_transformation(x, s=.2)]
train_set = IAMDataset(dataset_folder, subset='train', segmentation_level='fword', fixed_size=(64 + 1 * 32, 256 + 0 * 128), transforms=aug_transforms)  # (128, 1024))
test_set = IAMDataset(dataset_folder, subset='test', segmentation_level='fword', fixed_size=(64 + 1 * 32, 256 + 0 * 128))


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#form_val_set = IAMDataset(dataset_folder, subset='val', segmentation_level='form', fixed_size=None)
form_val_set = IAMDataset(dataset_folder, subset='test', segmentation_level='form', fixed_size=None)



ndisplay = 50
# use only Ns forms for validation - just to see if the process is going in the right direction
Ns = 10 # iam seg-free #docs eval

# augmentation using data sampler
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)


# hardcoded classes definition for English alphabet + numbers + punctuation

classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

# use reduced character set to be in line with KWS methods

reduced_charset = True

def reduced(istr):
    return ''.join([c if (c.isalnum() or c=='_' or c==' ') else '*' for c in istr.lower()])

if reduced_charset:
    classes = reduced(classes)
    nclasses = ''
    for c in classes:
        if c in nclasses:
            continue
        else:
            nclasses += c
    classes = nclasses

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

# CNN configuration
logger.info('Preparing Net...')

cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]
cnn_top = 128
cnn = CNN(cnn_cfg, cnn_top, len(classes))

if args.model_load_path is not None:
    cnn.load_state_dict(torch.load(args.model_load_path).state_dict())

cnn.cuda(args.gpu_id)


# define optimizer & scheduler

nlr = args.learning_rate
#restart_epochs = max_epochs #// 2
optimizer = torch.optim.AdamW(cnn.parameters(), nlr, weight_decay=0.00005)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, restart_epochs)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * max_epochs), int(.75 * max_epochs)], gamma=.1)


def train(epoch):

    cnn.train()
    optimizer.zero_grad()

    closs = []
    for iter_idx, (img, transcr, bbox) in enumerate(train_loader):

        if reduced_charset:
            transcr = [reduced(tt) for tt in transcr]


        img = Variable(img.cuda(gpu_id))

        ycnt, len_in, yctc = cnn(img, bbox)

        act_lens = torch.IntTensor(len_in)  # .to(img.device)

        labels = torch.IntTensor([cdict[c] for c in ''.join(transcr)]) #.to(img.device)
        label_lens = torch.IntTensor([len(t) for t in transcr]) #.to(img.device)

        loss1 = F.ctc_loss(F.log_softmax(yctc.cpu(), dim=2), labels, act_lens, label_lens,
                           zero_infinity=True, reduction='sum') / img.size(0)


        cnt_target = torch.zeros((ycnt.size(0), len(classes) - 1))
        for ii, tt in enumerate(transcr):
            for c in tt:
                if c in classes[1:-1]:
                    cnt_target[ii, cdict[c]-1] += 1.0
        # space indicator!
        cnt_target[:, -1] = 0.0

        cnt_target = cnt_target.float()
        mask = (cnt_target > 0).float()

        ycnt = ycnt.cpu()
        p = .5
        loss2 =  p * (mask * (ycnt - cnt_target)**2).sum()/mask.sum() + (1-p) * ((1 - mask) * (ycnt - cnt_target)**2).sum()/(1-mask).sum()
        loss2 = loss2 + 10. * torch.mean((ycnt.mean(-1) - cnt_target.mean(-1))**2)

        # help convergence by training htr only in the start
        if epoch < 2:
            loss_val = loss1
        else:
            loss_val = 1.0 * loss1 + 2 * loss2

        closs += [loss_val.data]

        loss_val.backward()

        # magnitude clipping
        torch.nn.utils.clip_grad_norm_(cnn.parameters(), .1)

        optimizer.step()
        optimizer.zero_grad()

        # mean runing errors??
        if iter_idx % ndisplay == (ndisplay-1):
            logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))
            logger.info('%f %f %f', loss1.item(), loss2.item(), eloss.item())
            #logger.info('lr: %f', optimizer.get_lr()[0])
            closs = []

            tst_img, tst_transcr, bbox = test_set.__getitem__(np.random.randint(test_set.__len__()))
            if reduced_charset:
                tst_transcr = reduced(tst_transcr)

            with torch.no_grad():
                ycnt, _, yctc, _ = cnn(Variable(tst_img.cuda(gpu_id)).unsqueeze(0), bbox.cuda(gpu_id).unsqueeze(0))


            print('orig:: ' + tst_transcr)
            tst_o = yctc
            tdec = tst_o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            print('gdec:: ' + ''.join([icdict[t] for t in tt]).replace('_', ''))

            cnt_str = []
            for c in tst_transcr:
                if c in classes[1:-1]:
                    ccnt = ycnt[0, cdict[c]-1]
                    cnt_str += ['/' + c + ' : ' + str(round(ccnt.item(), 3))]
            print(''.join(cnt_str))



import editdistance
# slow implementation
def test(epoch):
    cnn.eval()

    logger.info('Testing at epoch %d', epoch)

    tdecs = []
    transcrs = []
    for (img, transcr, bbox) in test_loader:
        img = Variable(img.cuda(gpu_id))

        ycnt, yctc = cnn(img, bbox)
        tdec = yctc.argmax(2).permute(1, 0).cpu().numpy().squeeze()
        tdecs += [tdec]
        transcrs += list(transcr)
    tdecs = np.concatenate(tdecs)

    cer, wer = [], []
    for tdec, transcr in zip(tdecs, transcrs):
        transcr = transcr.strip()
        if reduced_charset:
            transcr = reduced(transcr)

        tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
        dec_transcr = dec_transcr.strip()
        cer += [float(editdistance.eval(dec_transcr, transcr))/ len(transcr)]
        wer += [1 - float(transcr == dec_transcr)]

    logger.info('CER at epoch %d: %f', epoch, sum(cer) / len(cer))
    logger.info('WER at epoch %d: %f', epoch, sum(wer) / len(wer))

    cnn.train()

cnt = 0
logger.info('Training:')


@dataclass
class EvalArgs:
    clevels: int = 1
    cos_thres: float = 0.5
    ctc_thres: float = 3.5
    K: int = 50
    prob_thres: float = 0.05
    carea_ratio: float = 0.5
    ctc_mode: int = 2
    iou_mode: int = 2
    doc_scale: float = 1.0
    masked_form: bool = (dataset == 'iam')
    reduced_charset: bool = reduced_charset


eval_args = EvalArgs() #3.5)

best_map = 0
for epoch in range(1, max_epochs + 1):
    train(epoch)
    scheduler.step()

    if epoch % 2 == 0:
        test(epoch)
        tmp_map = seg_free_eval(form_val_set, cnn, classes, eval_args, Ns=Ns)
        if tmp_map > best_map:
            print('Saving net !!')
            torch.save(cnn.cpu(), args.model_save_path.replace('.pt', '_best.pt'))
            cnn.cuda(gpu_id)
            best_map = tmp_map

    if epoch % 5 == 0:
        torch.save(cnn.cpu(), args.model_save_path)
        cnn.cuda(gpu_id)


torch.save(cnn.cpu(), args.model_save_path)

