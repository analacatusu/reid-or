import numpy as np
import torch
from PIL import Image
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import cv2

'''
Old transformations, just SquarePad2 is needed
'''

class SquarePad2(object):

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        im = F.pad(image, padding, 0, 'constant')
        image = np.array(im)
        rd = np.random.randint(-50, -20, dtype=int)
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        if w > h:
            M = np.float32([[1, 0, 0], [0, 1, rd]])
        else:
            M = np.float32([[1, 0, rd], [0, 1, 0]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        im = Image.fromarray(image.astype('uint8'), 'RGB')
        return im

class SquarePad(object):

    def __call__(self, sample):
        image, id, im_path = sample['image'], sample['id'], sample['im_path']
        pil_image = Image.fromarray(image)
        w, h = pil_image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        im = F.pad(pil_image, padding, 0, 'constant')
        image = np.array(im)
        return {'image': image, 'id': id, 'im_path': im_path}


class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, id, im_path = sample['image'], sample['id'], sample['im_path']

        if isinstance(self.output_size, int):
            new_h = new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        resize = transforms.Resize((new_h, new_w))
        pil_image = Image.fromarray(image)
        img_resized = resize(pil_image)
        img = np.array(img_resized)
        return {'image': img, 'id': id, 'im_path': im_path}


class ToTensor(object):

    def __call__(self, sample):
        image, id, im_path = sample['image'], sample['id'], sample['im_path']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.astype(np.float32)),
                'id': torch.tensor([id]),
                'im_path': im_path}


class RandomVerticalFlip(object):

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        image, id, im_path = sample['image'], sample['id'], sample['im_path']
        image = Image.fromarray(image)
        flip = transforms.RandomVerticalFlip(self.probability)
        flipped_image = flip(image)
        # flipped_image.show()
        image_ndarray = np.array(flipped_image)
        return {'image': image_ndarray, 'id': id, 'im_path': im_path}


class RandomHorizontalFlip(object):

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        image, id, im_path = sample['image'], sample['id'], sample['im_path']
        image = Image.fromarray(image)
        flip = transforms.RandomHorizontalFlip(self.probability)
        flipped_image = flip(image)
        # flipped_image.show()
        image_ndarray = np.array(flipped_image)
        return {'image': image_ndarray, 'id': id, 'im_path': im_path}


class Normalize(object):

    def __init__(self, mean=None, std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, id, im_path = sample['image'], sample['id'], sample['im_path']
        normalize = transforms.Normalize(self.mean, self.std)
        norm_image = normalize(image)
        return {'image': norm_image, 'id': id, 'im_path': im_path}
