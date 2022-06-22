from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import option


def inverse_normalize(img):
    if option.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


'''def pytorch_normalize(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    #normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
     #                           std=[0.229, 0.224, 0.225])
    tensor = t.from_numpy(img)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = t.as_tensor(mean,)
    std = t.as_tensor(std)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
 
    return tensor.numpy()
    #return img
'''
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    tensor = t.from_numpy(img)
    dtype = tensor.dtype
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = t.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = t.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor.numpy()

def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    #img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    img = sktsf.resize(img, (C, 448, 448), mode='reflect',anti_aliasing=False)
    #img = img.transpose(1, 2, 0)
    #print(np.shape(img))
    # both the longer and shorter should be less than
    # max_size and min_size
    if option.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        #print("===============================")
        #print("raw img shape", img.shape, img)
        #print("raw bbox shape", bbox.shape, bbox)
      
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))
        #print("bbox after resize", bbox)
        # horizontally flip
        '''img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        print("bbox after flip", bbox)'''
        return img, bbox, label, scale


class Dataset:
    def __init__(self, option):
        self.option = option
        self.db = VOCBboxDataset(option.voc_data_dir)
        self.tsf = Transform(option.min_size, option.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        #print("after tsf bbox", bbox)
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, option, split='test', use_difficult=True):
        self.option = option
        self.db = VOCBboxDataset(option.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
