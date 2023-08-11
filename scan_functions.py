import numpy as np
from numba import jit

import torch

##############################################################################
# intersection metrics

@jit(nopython=True)
def intersections(bboxes1, bboxes2):

    inters = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    for i, bbox in enumerate(bboxes1):
        dx = - np.maximum(bbox[0], bboxes2[:, 0]) + np.minimum(bbox[2], bboxes2[:, 2])
        dy = - np.maximum(bbox[1], bboxes2[:, 1]) + np.minimum(bbox[3], bboxes2[:, 3])

        mask = (dx > 0) & (dy > 0)

        inters[i, :] = dx * dy * mask

    return inters

#@jit(nopython=True)
def intersection_metric(bboxes1, bboxes2, qids, return_no_intersection=False):

    inters = intersections(bboxes1, bboxes2)

    gt_areas = ((bboxes1[qids, 2] - bboxes1[qids, 0]) * (bboxes1[qids, 3] - bboxes1[qids, 1])).reshape(-1, 1)

    sinters = inters.sum(axis=0, keepdims=True)

    metric = inters[qids] / (gt_areas + sinters - inters[qids])

    if return_no_intersection:
        mask = sinters < 1e-2
        metric[mask] = -1

    return metric


##############################################################################
# binary search functions for finding the bbox size

@jit(nopython=True)
def find_scale(scale_cmap, i, j, max_step=100, target_scale=.5):  #.5):

    max_step = min(min(i+max_step, scale_cmap.shape[0]) -i, min(j+max_step, scale_cmap.shape[1])-j)

    l, h = 1, max_step
    while l <= h:
        step = l + (h - l) // 2

        pred = scale_cmap[i + step, j + step] + scale_cmap[i-1, j-1] \
               - scale_cmap[i + step, j-1] - scale_cmap[i-1, j + step]

        if pred > target_scale:
            h = step - 1
        else:
            l = step + 1

    return step + 1


@jit(nopython=True)
def precompute_scale(scale_cmap, area_thres=1.0, carea_ratio=.5):

    steps = np.zeros(scale_cmap.shape)
    for si in range(1, scale_cmap.shape[0]-1):
        for sj in range(1, scale_cmap.shape[1]-1):

            max_i_step = find_scale(scale_cmap, si, sj)

            area_sum = scale_cmap[si  - 1, sj -1] + \
                       scale_cmap[si + max_i_step, sj + max_i_step] - \
                       scale_cmap[si - 1, sj + max_i_step] - \
                       scale_cmap[si + max_i_step, sj - 1]

            if area_sum < 1 - area_thres or area_sum > 1 + area_thres:
               continue

            centered_area_sum = scale_cmap[si + max_i_step // 3 - 1, sj - 1] + \
                                scale_cmap[si + 2 * max_i_step // 3, sj + max_i_step] - \
                                scale_cmap[si + max_i_step // 3 - 1, sj + max_i_step] - \
                                scale_cmap[si + 2 * max_i_step // 3, sj - 1]

            if (centered_area_sum < carea_ratio *area_sum):
                continue

            steps[si, sj] = max_i_step

    return steps


@jit(nopython=True)
def find_bbox(scale_cmap, target, si, sj, istep, max_rj=100, tr=.75):

    l, h = 1, min(scale_cmap.shape[1], sj + max_rj) - sj

    while l <= h:
        step = l + (h - l) // 2

        pred = scale_cmap[si + istep, sj + step] + scale_cmap[si-1, sj-1] \
               - scale_cmap[si + istep, sj-1] - scale_cmap[si-1,  sj + step]

        if pred > tr * target:
            h = step - 1
        else:
            l = step + 1

    return step + 1

##############################################################

# first scanning step - provides a list of candidate bboxes
# integral maps are used for speed - binary search is used for finding the bbox size

@jit(nopython=True)
def scan_page(scale_cmap, steps, valid_map, starget, prob_thres=.05):

    bboxes = [np.int(x) for x in range(0)]
    carea_ratio = [np.float(x) for x in range(0)]
    for si in range(1, scale_cmap.shape[0]-1, 1):
        for sj in range(1, scale_cmap.shape[1]-1, 1):

            istep = int(steps[si, sj])
            if istep == 0:
                continue

            #if valid_map[si + istep // 2, sj + istep//4] < prob_thres:
            if valid_map[si + istep // 2, sj + 1] < prob_thres:
                continue

            max_jstep = int(2 * istep * starget)

            jstep = find_bbox(scale_cmap, starget, si, sj, istep, max_jstep)

            area_sum = scale_cmap[si  - 1, sj -1] + \
                       scale_cmap[si + istep, sj + jstep] - \
                       scale_cmap[si - 1, sj + jstep] - \
                       scale_cmap[si + istep, sj - 1]

            centered_area_sum = scale_cmap[si + istep // 3 - 1, sj - 1] + \
                                scale_cmap[si + 2 * istep // 3, sj + jstep] - \
                                scale_cmap[si + istep // 3 - 1, sj + jstep] - \
                                scale_cmap[si + 2 * istep // 3, sj - 1]

            if (centered_area_sum < .5 * starget) or (area_sum < .75 * starget) or (centered_area_sum < .5 *area_sum):
                continue
            

            carea_ratio += [centered_area_sum / area_sum]

            bboxes += [sj, si, min(scale_cmap.shape[1]-1, sj + jstep), min(scale_cmap.shape[0]-1, si + istep)]

    return np.asarray(bboxes), np.asarray(carea_ratio)


##############################################################

# function that computes the CNN output and auxiliary maps that assist the scanning process

def generate_maps(img, cnn, mask=None, carea_ratio=.5):

    img = img.view([1, 1] + list(img.size()))

    # compute CNN output
    with torch.no_grad():
        yc, s = cnn(img, reduce=False)

    # use a mask for constraining area of search !!
    if mask is not None:
        mask = mask.view([1, 1] + list(mask.size()))
        s = s * (torch.nn.functional.interpolate(mask, size=[s.size(2), s.size(3)], mode='bilinear') > 1e-3).float()

    # character probs map
    rmap = torch.nn.functional.softmax(yc, 1) * s

    mask = 0 #-10 * (1 - s)

    # precomputed auxiliary maps
    ctc_map = torch.nn.functional.softmax(torch.nn.functional.max_pool2d(1.0 * yc + mask, [3, 1], 1, [1, 0]), 1)[0].permute(1, 2, 0).cpu().detach().numpy()
    valid_map = torch.nn.functional.max_pool2d(torch.nn.functional.softmax(1.0 * yc, 1), [3, 3], 1, [1, 1])[0].permute(1, 2, 0).cpu().detach().numpy()

    rmap = rmap[0].permute(1, 2, 0).cpu().detach().numpy()
    # integral map using cumsum
    cmap = rmap[:, :, 1:].cumsum(axis=0).cumsum(axis=1)

    # scale integral map
    scale_cmap = cmap.sum(axis=-1) #scale_map.cumsum(axis=0).cumsum(axis=1)

    # precompute character size for every pixel
    steps = precompute_scale(scale_cmap, carea_ratio=carea_ratio)

    return ctc_map, cmap, scale_cmap, valid_map, steps

##############################################################

# simple function that computes pyramidal counting representation on query strinfs

def phoc_like(query , cdict, levels=1):

    N = len(query)
    vinds = np.asarray([cdict[c] - 1 for c in query])

    descriptor = []
    for K in range(levels):

        sep = np.linspace(0, N - 1, 2 + K)
        sc = .5 * (sep[1:] + sep[:-1])

        ww = np.tile(np.arange(0, N).reshape(1, -1), (K + 1, 1))

        ss = 1.0 * sc[0]
        ww = np.exp(-(ww - sc.reshape(-1, 1)) ** 2 / (2 * (1.0 * ss) ** 2))
        ww = ww / ww.sum(0)[None, :]

        cnt_targets = np.zeros((ww.shape[0], len(cdict)-1))
        for i, vind in enumerate(vinds):
            cnt_targets[:, vind] += ww[:, i]

        descriptor += [cnt_targets.reshape(-1)]

    return np.concatenate(descriptor)

# PHOC-like function that computes pyramidal counting representation of an area of the image

@jit(nopython=True)
def pyramidal_counting(cmap, si, sj, ei, ej, levels):

    pred = np.zeros(levels * (levels + 1) * cmap.shape[-1] // 2)
    cnt = 0
    for level in range(levels):
        div = level + 1
        for il in range(level + 1):
            tsj, tej = sj + il * (ej - sj) // div, min(cmap.shape[1] - 1, sj + (il + 1) * (ej - sj) // div)
            pred[cnt * cmap.shape[-1]: (cnt + 1) * cmap.shape[-1]] = cmap[ei, tej] + cmap[si - 1, tsj - 1] - cmap[
                ei, tsj - 1] - cmap[si - 1, tej]
            cnt += 1

    return pred #np.asarray(pred) #np.concatenate(pred)

@jit(nopython=True)
def cos_scores(cmap, bboxes, cnt_target, levels=1, phoc_activate=False):
    #cnt_target = cnt_target.astype(np.float32) # numba compatibility

    if phoc_activate:
        cnt_target = np.clip(cnt_target, 0, 1)

    scores = np.zeros(bboxes.shape[0])
    #valid_inds = [np.int(x) for x in range(0)]
    tnorm = np.linalg.norm(cnt_target)
    starget = np.sum(cnt_target)
    for i, bbox in enumerate(bboxes):
        si, sj, ei, ej = bbox[1], bbox[0], bbox[3], bbox[2]

        pred = pyramidal_counting(cmap, si, sj, ei, ej, levels)

        if phoc_activate:
            pred = np.clip(pred, 0, 1)

        tscore = np.dot(pred, cnt_target)
        if tscore < .5 * starget and tscore > 1.5 * starget:
            continue

        tscore = tscore /  (1e-10 + tnorm * np.linalg.norm(pred))
        scores[i] = tscore #- .1 * pred[-1] # penalize space !!

    return bboxes, scores

##############################################################

# non maximum suppression function

@jit(nopython=True)
def np_nms(dets, scores, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maximum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nms_update(bboxes, scores, nms_iou, k):

    valid_indices = np_nms(bboxes, scores, nms_iou)

    if len(valid_indices) > k:
        valid_indices = valid_indices[:k]

    bboxes = bboxes[valid_indices]
    scores = scores[valid_indices]

    return bboxes, scores

##############################################################

# basic reimplementation of ctc in jit for speed
# has some extra properties like:
# hard mode: if true acts like viterbi decoding, else acts as the typical soft ctc process
# min mode: if true returns the index of the best score of the last query character

@jit(nopython=True)
def myctc(log_probs, labels, hard_mode=False, min_mode=True):
    extended_labels = [0]
    for label in labels:
        extended_labels += [label, 0]

    # logD matrix
    D = -np.inf * np.ones((len(extended_labels), log_probs.shape[0]))
    D[0, :] = log_probs[:, extended_labels[0]]
    D[1, 0] = log_probs[0, extended_labels[1]]

    for t in range(1, log_probs.shape[0]):
        D[0, t] += D[0, t-1]

    for t in range(1, log_probs.shape[0]):
        if hard_mode:
            D[1, t] = log_probs[t, extended_labels[1]] + max(D[1, t - 1], D[0, t - 1])
        else:
            D[1, t] = log_probs[t, extended_labels[1]] + np.logaddexp(D[1, t-1], D[0, t-1])

    for t in range(1, log_probs.shape[0]):
        for c in range(2, len(extended_labels)):
            if hard_mode:
                tmp = max(D[c, t - 1], D[c - 1, t - 1])
            else:
                tmp = np.logaddexp(D[c, t - 1], D[c-1, t - 1])
            if (extended_labels[c] == 0) or (extended_labels[c] == extended_labels[c-2]):
                D[c, t] = log_probs[t, extended_labels[c]] + tmp
            else:
                if hard_mode:
                    D[c, t] = log_probs[t, extended_labels[c]] + max(tmp, D[c - 2, t - 1])
                else:
                    D[c, t] = log_probs[t, extended_labels[c]] + np.logaddexp(tmp, D[c-2, t - 1])

    # traceback -  assume monotone last section
    if hard_mode:
        fscores = -np.maximum(D[-1, :], D[-2, :])
    else:
        fscores = -np.logaddexp(D[-1, :], D[-2, :])


    if min_mode:
        offset = min(np.argmin(fscores) + 2, D.shape[1] - 1)
    else:
        offset = D.shape[1] - 1  #

    return fscores[offset]/len(labels), offset


# compute ctc scores for every candidate bbox
# ctc_thres is the threshold for the ctc score that prunes candidates
# ctc_mode: 0: typical ctc, 1: ctc with min mode, 2: ctc with min mode and backward pass

@jit(nopython=True)
def ctc_scores(rmap, bboxes, labels, ctc_thres, ctc_mode=2):

    offset = 5

    nbboxes, nscores = [np.int(x) for x in range(0)], [np.float(x) for x in range(0)]
    for bbox in bboxes:
        sj, si, ej, ei = bbox
        sj = max(sj, 0)
        si = max(si, 0)
        ei = min(max(ei, si + 1), rmap.shape[0])
        ej = min(max(ej, sj + 1), rmap.shape[1])

        tmap = rmap[(si + ei + 1) // 2, sj:min(ej + offset + 1, rmap.shape[1])]
        #tmap = np.maximum(rmap[(si + ei + 1) // 2, sj:ej + 1], rmap[(si + ei + 1) // 2 - 1, sj:ej + 1])
        tmap = np.log(1e-10 + tmap)
        #tmap = np.log(tmap + 1e-10) - np.log(tmap.sum(axis=-1, keepdims=True) + 1e-10)

        if ctc_mode == 0:
            ctc_score, _ = myctc(tmap, labels, hard_mode=False, min_mode=False)
        elif ctc_mode == 1 or ctc_mode == 2:
            ctc_score, jstep = myctc(tmap, labels, hard_mode=False, min_mode=True)
            ej = max(sj + jstep, sj + 1)

            if ctc_mode == 2:
                ## backward!!
                tmap = rmap[(si + ei + 1) // 2, max(sj - offset, 0):ej +1]
                #tmap = np.maximum(rmap[(si + ei + 1) // 2, sj:ej + 1], rmap[(si + ei + 1) // 2 - 1, sj:ej + 1])
                tmap = np.log(1e-10 + tmap)

                ctc_score, bjstep = myctc(tmap[::-1], labels[::-1], hard_mode=False, min_mode=True)
                sj = ej - bjstep - 1
        else:
            print('not valid ctc mode!!')


        if ctc_score > ctc_thres:
            continue
        else:
            nbboxes += [sj, si, ej, ei]
            nscores += [ctc_score]

    bboxes, scores = np.asarray(nbboxes).reshape(-1, 4), np.asarray(nscores)

    return bboxes, scores


##############################################################

# main function for keyword spotting

def form_kws(rmap, cmap, scale_cmap, valid_map, steps, query, query_desc, classes, k=30, nms_iou=.1, cos_thres=.66, clevels=1, ctc_thres=None, ctc_mode=2, prob_thres=.01):

    cdict = {c: i for i, c in enumerate(classes)}

    cnt_target = np.zeros(len(classes) - 1)
    for c in query:
        if c in classes[1:-1]:
            cnt_target[cdict[c] - 1] += 1.0 #.8 #1.0

    starget = cnt_target.sum()

    # !!!!!!!!!!!!!!
    tvmap = valid_map[..., cdict[query[0]]]

    bboxes, carea_ratio = scan_page(scale_cmap, steps, tvmap, starget, prob_thres)
    bboxes = np.asarray(bboxes).reshape(-1, 4)
    scores = np.zeros(bboxes.shape[0])

    #enlarge area of interest by a small margin
    dilate_r = 0.1
    if bboxes.shape[0] > 0:
        bboxes[:, 0] = bboxes[:, 0] - dilate_r * (bboxes[:, 2] - bboxes[:, 0])
        bboxes[:, 2] = bboxes[:, 2] + dilate_r * (bboxes[:, 2] - bboxes[:, 0])

        bboxes[:, 1] = bboxes[:, 1] - 2
        bboxes[:, 3] = bboxes[:, 3] + 2

    # first similarity matching step - pyramidal counting
    if bboxes.shape[0] > 0:
        bboxes, scores = cos_scores(cmap, bboxes, query_desc, levels=clevels)
        mask = scores > cos_thres

        bboxes = bboxes[mask]
        scores = scores[mask]

        # non maxima suppression
        if bboxes.shape[0] > 0:
            bboxes, scores = nms_update(bboxes, scores, nms_iou, k)

    # second similarity matching step - ctc scoring
    if bboxes.shape[0] > 0:

        if ctc_thres is not None:

            bboxes, scores = ctc_scores(rmap, bboxes, np.asarray([cdict[c] for c in query]), ctc_thres, ctc_mode)

            if bboxes.shape[0] > 0:
                # non maxima suppression
                bboxes, scores = nms_update(bboxes, -scores, .01, k)
                scores = - scores
        else:
            scores = 1 - scores # cosine similarity to cosine distance

    #if bboxes.shape[0] > 0:
    #    bboxes[:, 1] = bboxes[:, 1] - 1
    #    bboxes[:, 3] = bboxes[:, 3] + 1

    return bboxes, scores


