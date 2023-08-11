import numpy as np
import logging

import torch.cuda

#from eval_config import *

import time

from utils.auxilary_functions import average_precision

from torchvision.ops.boxes import box_iou
import torch.nn.functional as F

from scan_functions import form_kws, generate_maps, intersection_metric, phoc_like

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def reduced(istr):
    return ''.join([c if (c.isalnum() or c=='_' or c==' ') else '*' for c in istr.lower()])

def seg_free_eval(form_test_set, cnn, classes, args, Ns=None, eval_multiple_thres=False):

    device = next(cnn.parameters()).device
    stopwords = form_test_set.stopwords
    reduced_charset = args.reduced_charset

    #classes = args.classes
    if reduced_charset:
        classes = reduced(classes)
        nclasses = ''
        for c in classes:
            if c in nclasses:
                continue
            else:
                nclasses += c
        classes = nclasses

    cdict = {c: i for i, c in enumerate(classes)}

    N = form_test_set.__len__()
    if Ns is not None and Ns < N:
        N = Ns

    transcrs = []
    for i in range(N):
        bboxes = form_test_set.__getitem__(i)[-1]
        transcrs += [tt[1] for tt in bboxes]

    if reduced_charset:
        transcrs = [reduced(tt) for tt in transcrs]

    uwords = np.unique(transcrs)
    udict = {w: i for i, w in enumerate(uwords)}
    lbls = np.asarray([udict[w] for w in transcrs])
    cnts = np.bincount(lbls)


    if reduced_charset:
        queries = [w for w in uwords if w not in stopwords and cnts[udict[w]] >= 1 and len(w) > 1 and '*' not in w]
    else:
        queries = [w for w in uwords if w not in stopwords and cnts[udict[w]] >= 1 and len(w) > 1]

    # print(queries)

    # threshold for returning most relevant words!
    doc_scale = args.doc_scale
    cos_thres = args.cos_thres  # .5
    ctc_thres = args.ctc_thres  # None #3.5
    if ctc_thres < 0:
        ctc_thres = None
    ctc_mode = args.ctc_mode  # False
    K = args.K

    levels = args.clevels
    iou_mode = args.iou_mode  # 0: typical IoU, 1: per-axis IOU, 2: intersection

    masked_form = args.masked_form

    query_descs = [phoc_like(query, cdict, levels) for query in queries]
    retrieval_dict = {query: ([], [], 0) for query in queries}

    tsum = 0
    for i in range(N):

        img, _, bboxes = form_test_set.__getitem__(i)

        if doc_scale != 1.0:
            img = F.interpolate(img.unsqueeze(0), scale_factor=doc_scale, mode='bilinear')[0]

        img = img[0]

        tbboxes, ttranscrs = [], []
        for bbox in bboxes:
            tbbox = doc_scale * bbox[0]
            si, ei = tbbox[1], min(img.shape[0], tbbox[1] + tbbox[3])
            sj, ej = tbbox[0], min(img.shape[1], tbbox[0] + tbbox[2])

            tbboxes += [[int(sj), int(ej), int(si), int(ei)]]
            if reduced_charset:
                ttranscrs += [reduced(bbox[1].strip())]
            else:
                ttranscrs += [bbox[1]]

        if masked_form:
            mask = np.zeros(img.shape)
            for bbox in tbboxes:
                sj, ej, si, ei = bbox
                mask[si:ei, sj:ej] = 1

            img *= torch.Tensor(mask)

        img = img.to(device)
        rmap, cmap, scale_cmap, valid_map, steps = generate_maps(img, cnn, carea_ratio=args.carea_ratio)

        # prepare usage of torcvision box iou
        tbboxes = torch.Tensor(np.asarray(tbboxes)[:, [0, 2, 1, 3]])

        tnow = time.time()

        for j, query in enumerate(queries):  # [100:400]:
            predicted_bboxes, predicted_scores = form_kws(rmap, cmap, scale_cmap, valid_map, steps, query,
                                                          query_descs[j], classes,
                                                          k=K, clevels=levels, cos_thres=cos_thres, ctc_thres=ctc_thres,
                                                          ctc_mode=ctc_mode, prob_thres=args.prob_thres)

            qidxs = [ii for ii, tt in enumerate(ttranscrs) if query == tt]
            realv = len(qidxs)


            if len(predicted_scores) == 0:
                retrieval_dict[query] = (
                    retrieval_dict[query][0],
                    retrieval_dict[query][1],
                    retrieval_dict[query][2] + realv
                )
                continue

            bdscale = 8  # * doc_scale

            predicted_bboxes = torch.Tensor(np.asarray(predicted_bboxes)) #[:, [0, 2, 1, 3]] / bdscale
            matches = np.zeros(len(predicted_scores))

            if realv > 0:
                if iou_mode == 0 or iou_mode == 1:
                    iou_scores, iou_args = box_iou(tbboxes[qidxs].view(-1, 4) / bdscale,
                                                   predicted_bboxes.view(-1, 4)).max(dim=-1)

                elif iou_mode == 2:
                    iou_scores = intersection_metric(tbboxes.view(-1, 4).numpy() / bdscale,
                                                     predicted_bboxes.view(-1, 4).numpy(),
                                                     qidxs)

                    iou_args = iou_scores.argmax(axis=-1)
                    iou_scores = iou_scores.max(axis=-1)
                else:
                    print('not valid iou mode')
                    exit(0)

                if len(iou_args) >= 1:
                    _, ainds = np.unique(iou_args, return_index=True)
                    iou_scores, iou_args = iou_scores[ainds], iou_args[ainds]

                    if iou_mode == 1:
                        xthres = .1
                        if sum(iou_scores > xthres) > 0:
                            # then y-axis
                            gb, pb = tbboxes[qidxs].view(-1, 4).clone() / bdscale, predicted_bboxes.view(-1, 4).clone()
                            gb = gb[torch.LongTensor(ainds)[iou_scores > xthres]]
                            pb = pb[iou_args[iou_scores > xthres]]

                            gb[:, 1] = 0
                            pb[:, 1] = 0
                            gb[:, 3] = 10
                            pb[:, 3] = 10

                            y_iou_scores = torch.diagonal(box_iou(gb, pb))
                            matches[iou_args[iou_scores > xthres]] = y_iou_scores
                    else:
                        matches[iou_args] = iou_scores

            retrieval_dict[query] = (
                np.concatenate([retrieval_dict[query][0], matches]),
                np.concatenate([retrieval_dict[query][1], np.asarray(predicted_scores)]),
                retrieval_dict[query][2] + realv
            )

        tsum += (time.time() - tnow)

        if i % 10 == 0:
            print('time @ ' + str(i) + ' : ' + str(tsum / ((i + 1) * len(queries))))

    print('overall time: ' + str(tsum / (N * len(queries))))

    if eval_multiple_thres:
        for iou_thres in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # calculate AP for each query!
            aps = []
            for key, (matches, scores, real_gt) in retrieval_dict.items():
                if real_gt > 0:
                    if len(matches) > 0:
                        sorted_inds = np.argsort(scores)
                        matches = matches[sorted_inds]
                        ap = average_precision(matches > iou_thres, real_gt)
                    else:
                        ap = 0
                else:
                    # print('huh? query: ' + str(key))
                    continue

                aps += [ap]

            map_metric = np.mean(aps)
            print('MAP @ ' + str(iou_thres) + ' IOU: ' + str(map_metric))
    else:
        # check only 25% for validation!
        iou_thres = 0.25
        aps = []
        for key, (matches, scores, real_gt) in retrieval_dict.items():
            if real_gt > 0:
                if len(matches) > 0:
                    sorted_inds = np.argsort(scores)
                    matches = matches[sorted_inds]
                    ap = average_precision(matches > iou_thres, real_gt)
                else:
                    ap = 0
            else:
                # print('huh? query: ' + str(key))
                continue

            aps += [ap]

        map_metric = np.mean(aps)
        print('MAP @ ' + str(iou_thres) + ' IOU: ' + str(map_metric))

    return map_metric