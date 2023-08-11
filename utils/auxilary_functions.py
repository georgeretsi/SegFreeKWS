import numpy as np

from skimage.transform import resize
from skimage import io as img_io
from skimage.color import rgb2gray

import torch
import torch.nn.functional as F
from scipy import interpolate
import cv2

def smooth_rand(sh, sw, e=5.0, s=1.0, fast=True):

    xx, yy = np.meshgrid(range(sw), range(sh))
    if fast:
        N = int(sh * sw * np.random.uniform(.15, .25))
        x = np.random.uniform(0, sw-1, N)
        y = np.random.uniform(0, sh-1, N)
        a = np.random.uniform(0, 1, N)
        rbf = interpolate.Rbf(x, y, a, epsilon=e, smooth=s)
    else:
        a = np.random.uniform(0, 1, (sh, sw))
        rbf = interpolate.Rbf(xx, yy, a, epsilon=e, smooth=s)
    nz = rbf(xx, yy)
    #nz = np.clip(nz, 0, 1)

    return nz


def intensity(img, dscale=8):

    h, w = img.shape[0], img.shape[1]
    sh, sw = h / dscale, w / dscale
    m = smooth_rand(sh, sw, 2, .1)
    mask = image_resize(1 + 2.0 * (m - .5), h, w)
    img = img * mask

    return img

def affine_transformation(img, m=1.0, s=.2, border_value=None):
    h, w = img.shape[0], img.shape[1]
    src_point = np.float32([[w / 2.0, h / 3.0],
                            [2 * w / 3.0, 2 * h / 3.0],
                            [w / 3.0, 2 * h / 3.0]])
    random_shift = m + np.random.uniform(-1.0, 1.0, size=(3,2)) * s
    dst_point = src_point * random_shift.astype(np.float32)
    transform = cv2.getAffineTransform(src_point, dst_point)
    if border_value is None:
        border_value = np.median(img)
    warped_img = cv2.warpAffine(img, transform, dsize=(w, h), borderValue=float(border_value))
    return warped_img

def image_resize(img, height=None, width=None):

    if height is not None and width is None:
        scale = float(height) / float(img.shape[0])
        width = int(scale*img.shape[1])

    if width is not None and height is None:
        scale = float(width) / float(img.shape[1])
        height = int(scale*img.shape[0])

    img = resize(image=img, output_shape=(height, width)).astype(np.float32)

    return img


def centered(word_img, tsize, centering=(.5, .5), border_value=None):

    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if len(word_img.shape) == 2:
        if border_value is None:
            border_value = np.median(word_img)

        word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw), 'constant', constant_values=border_value)

    elif len(word_img.shape) == 3:
        output = np.zeros((height, width, word_img.shape[-1]))
        if border_value is None:
            border_value = np.median(word_img[...,0])
        output[..., 0]= np.pad(word_img[ys:ye, xs:xe, 0], (padh, padw), 'constant', constant_values=border_value)
        output[..., 1] = np.pad(word_img[ys:ye, xs:xe, 1], (padh, padw), 'constant', constant_values=0)

        word_img = output

    return word_img



# check this
def average_precision(ret_vec_relevance, gt_relevance_num=None):
    '''
    Computes the average precision from a list of relevance items

    Params:
        ret_vec_relevance: A 1-D numpy array containing ground truth (gt)
            relevance values
        gt_relevance_num: Number of relevant items in the data set
            (with respect to the ground truth)
            If None, the average precision is calculated wrt the number of
            relevant items in the retrieval list (ret_vec_relevance)

    Returns:
        The average precision for the given relevance vector.
    '''
    if ret_vec_relevance.ndim != 1:
        raise ValueError('Invalid ret_vec_relevance shape')

    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range

    if gt_relevance_num is None:
        n_relevance = ret_vec_relevance.sum()
    else:
        n_relevance = gt_relevance_num

    if n_relevance > 0:
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    return ret_vec_ap