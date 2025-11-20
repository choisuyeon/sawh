"""
Data loader for the CGH dataset.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC). In a nutshell:
    - The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    - The material is provided as-is, with no warranties whatsoever.
    - If you publish any code, data, or scientific work based on this, please cite our work.

Article:
S. Choi, C. Jang, D. Lanman, G. Wetzstein,
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue",
Nature Photonics, 2025
"""

import os
import random
import logging
import itertools

import numpy as np
import torch
import cv2
from imageio import imread
from skimage.transform import resize
from torch.utils.data import Dataset
import utils.utils as utils
from utils.load_unity_light_field import load_unity_light_field
from torchvision.transforms.functional import resize as resize_tensor

import hardware.utils_mems as utils_mems
from hardware.utils_mems import AngleConversion

def load_data_loader(cfg, dev):
    return TargetLoader(cfg, dev)


def get_image_filenames(dir, focuses=None):
    """Returns all files in the input directory dir that are images."""
    image_types = (
        'jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif', 'exr', 'dpt', 'hdf5'
    )
    if isinstance(dir, str):
        files = os.listdir(dir)
        exts = [os.path.splitext(f)[1] for f in files]
        if focuses is not None:
            images = [
                os.path.join(dir, f)
                for e, f in zip(exts, files)
                if e[1:] in image_types
                and int(os.path.splitext(f)[0].split('_')[-1]) in focuses
            ]
        else:
            images = [
                os.path.join(dir, f)
                for e, f in zip(exts, files)
                if e[1:] in image_types
            ]
        return images
    elif isinstance(dir, list):
        # Support multiple directories (randomly shuffle all)
        images = []
        for folder in dir:
            files = os.listdir(folder)
            exts = [os.path.splitext(f)[1] for f in files]
            images_in_folder = [
                os.path.join(folder, f)
                for e, f in zip(exts, files)
                if e[1:] in image_types
            ]
            images.extend(images_in_folder)
        return images
    return []


def resize_keep_aspect(image, target_res, pad=False, lf=False, pytorch=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping.

    Args:
        image: a 3d array with dims [channel, height, width]
        target_res: [height, width]
        pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # Find the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions.
    resized_res = (
        int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
        int(np.ceil(im_res[0] * target_res[1] / target_res[0])),
    )

    # Only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions.
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False, lf=lf)

    # Switch to numpy channel dim convention, resize, switch back
    if lf or pytorch:
        image = resize_tensor(image, target_res)
        return image
    else:
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res, pytorch=True, lf=False):
    """Pads with 0 and crops as needed to force image to be target_res.

    Args:
        image: an array with dims [..., channel, height, width]
        target_res: [height, width]
    """
    return utils.crop_image(
        utils.pad_image(image, target_res, pytorch=pytorch, stacked_complex=False, lf=lf),
        target_res, pytorch=pytorch, stacked_complex=False, lf=lf
    )


class TargetLoader(Dataset):
    def __init__(self, cfg, dev=torch.device('cuda')):
        self.cfg = cfg
        self.dev = dev
        self.data_path = cfg.cgh.data_path
        self.target_type = cfg.cgh.data_type
        self.target_names = []
        if self.target_type == '2d':
            self.target_names = get_image_filenames(self.data_path)
        else:
            self.target_names = [
                os.path.join(self.data_path, cfg.cgh.lf_data.data_path)
            ]
        if self.target_type == '2d':
            self.crop_to_roi = cfg.cgh.crop_to_roi
        else:
            self.crop_to_roi = cfg.cgh.lf_data.crop_to_roi  # if False, resize
        self.order = [(i,) for i in range(len(self.target_names))]
        self.channel = cfg.model.channel
        self.linear_target = cfg.cgh.linear_target
        self.shuffle = cfg.cgh.data_shuffle
        self.data_mask_target = cfg.cgh.data_mask_target
        self.roi_res = cfg.cgh.roi_res
        self.flipud = cfg.cgh.target_flipud

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.idx < len(self.order):
            idx = self.order[self.idx]
            self.idx += 1
            return self.__getitem__(*idx)
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if self.target_type == '2d':
            return self.load_image(idx)
        if self.target_type == '4d':
            return self.load_lf(idx)
        raise ValueError(f"Unknown target_type: {self.target_type}")

    def load_image(self, filenum, *augmentation_states):
        target_name = self.target_names[filenum]
        target_id = os.path.basename(os.path.splitext(target_name)[0])
        im = imread(target_name)

        if len(im.shape) < 3:
            # augment channels for grayscale images
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

        # linearize intensity and convert to amplitude
        if not self.linear_target:
            im = utils.srgb_gamma2lin(im)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # normalize resolution
        if self.cfg.cgh.image_res is not None:
            if self.crop_to_roi:
                im = pad_crop_to_res(im, self.cfg.cgh.roi_res)
            else:
                im = resize_keep_aspect(im, self.cfg.cgh.roi_res)
            im = pad_crop_to_res(im, self.cfg.cgh.image_res)
        else:
            if getattr(self.cfg, "resize_image_res", None) is not None:
                im = resize_keep_aspect(im, self.cfg.resize_image_res)

        im = torch.from_numpy(im).float().to(self.dev)

        data = {
            'target_amp': im.unsqueeze(0),
            'target_id': target_id
        }
        return data

    def load_lf(self, idx):
        folder_path = self.target_names[idx]

        def get_lf_params(cfg):
            logging.info("DataLoader: getting LF params ...")
            start_position_idx, end_position_idx = utils.get_start_and_end_position_idx(
                cfg.cgh.steered_angle, cfg.cgh, cfg.model
            )
            lf_params = {
                'feature_size': (cfg.model.pixel_pitch, cfg.model.pixel_pitch),
                'lf_stride': 1,
                'lf_start_idx': start_position_idx,
                'lf_end_idx': end_position_idx,
                'num_views_to_load': tuple(
                    end - start + 1
                    for start, end in zip(start_position_idx, end_position_idx)
                )
            }
            return lf_params

        lf_params = get_lf_params(self.cfg)
        lf, depth = load_unity_light_field(
            folder_path,
            self.cfg.cgh.focal_length_eyepiece,
            channel=self.channel,
            lf_params=lf_params,
            loadOnlyCentralView=False
        )
        if self.cfg.cgh.image_res is not None:
            if self.crop_to_roi:
                lf = pad_crop_to_res(lf, self.cfg.cgh.roi_res)
            else:
                lf = resize_keep_aspect(lf, self.cfg.cgh.roi_res, lf=True)
            lf = pad_crop_to_res(lf, self.cfg.cgh.image_res)

        lf = torch.tensor(lf, dtype=torch.float32, device=self.dev)
        lf = lf.permute(2, 3, 0, 1).unsqueeze(0).sqrt()

        if self.flipud:
            logging.info('flipping LF u,v ... ud')
            lf = lf.flip(dims=[-2])

        data = {
            'target_amp': lf.unsqueeze(0),
            'target_id': self.cfg.cgh.lf_data.name
        }
        return data

    def augment_vert(self, image=None, flip=False):
        """Augment data with vertical flip."""
        if image is None:
            return True, False  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        """Augment data with horizontal flip."""
        if image is None:
            return True, False  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image


def load_angles_to_train(angle_train_path):
    """Load a list of angles (y, x) from a txt file."""
    with open(angle_train_path, 'r') as f:
        angles_step = [tuple(map(float, line.strip().split(','))) for line in f]
    return angles_step


class PairsLoader(torch.utils.data.IterableDataset):
    def __init__(
        self, cfg, dataset='train', num_phases=None, dev=torch.device('cuda')
    ):
        self.angle_conversion = AngleConversion(None)  # convert step to rad
        self.angle_train_path = cfg.angle_train_path  # list of angles in step
        self.angles_to_train = load_angles_to_train(self.angle_train_path)
        self.dev = dev
        self.phase_path = cfg.dataset.data_phase_path
        self.captured_path = cfg.dataset.data_captured_path  # actually not being used

        self.im_names = []
        self.im_names.extend(get_image_filenames(self.phase_path))
        logging.info(f'PairsLoader: total number of phases:{len(self.im_names)}, selecting num_phases:{num_phases}')
        self.im_names.sort()
        random.seed(52)
        random.shuffle(self.im_names)
        if num_phases is not None:
            if dataset == 'train':
                self.im_names = self.im_names[:num_phases]
            else:
                self.im_names = self.im_names[-num_phases:]  # from back for validation
        else:
            if dataset == 'train':
                self.im_names = self.im_names[:int(len(self.im_names) * 0.9)]
            else:
                self.im_names = self.im_names[-int(len(self.im_names) * 0.1):]  # from back for validation

        self.batch_size = cfg.batch_size
        self.shuffle = cfg.data_shuffle
        self.phase_angle_pairs = list(itertools.product(self.im_names, self.angles_to_train))

        # create list of image IDs with augmentation state
        self.order = [(i,) for i in range(len(self.phase_angle_pairs))]
        self.image_domain = cfg.dataset.captured_domain  # 'amp' or 'lin'
        self.flip_training_images = cfg.dataset.flip_training_images

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.ind < len(self.order):
            phase_idx = self.order[self.ind]
            self.ind += 1
            return self.load_pair_all(phase_idx[0])
        else:
            raise StopIteration

    def load_pair_all(self, filenum):
        """
        Load a pair of phase and captured image.

        Args:
            filenum: the index of the pair to load

        Returns:
            phase_im: the phase image
            captured_amp: the captured image
            angle_rad: the angle in radians
        """
        phase_path, angle_step = self.phase_angle_pairs[filenum]

        # get dirname and filename
        captured_path = os.path.join(
            os.path.dirname(os.path.dirname(phase_path)), 'captured'
        )
        phase_filename = os.path.basename(phase_path)

        # load_phase
        phase_im_enc = cv2.imread(phase_path, cv2.IMREAD_UNCHANGED)
        im = (1 - phase_im_enc / np.iinfo(np.uint8).max) * 2 * np.pi - np.pi
        phase_im = torch.tensor(im, dtype=torch.float32).unsqueeze(0)
        if getattr(self, 'flip_training_images', False):
            # Flip horizontally (dim=-1)
            phase_im = torch.flip(phase_im, dims=[-1])
            # Flip vertically (dim=-2)
            phase_im = torch.flip(phase_im, dims=[-2])

        # convert MEMS step to angle in radians
        angle_rad = self.angle_conversion(angle_step)

        # captured_filename
        captured_filename = os.path.join(
            captured_path,
            f'{angle_step[0]:.4f}_{angle_step[1]:.4f}',
            phase_filename
        )

        captured_intensity = cv2.imread(captured_filename, cv2.IMREAD_UNCHANGED)
        captured_intensity = utils.im2float(captured_intensity)
        captured_intensity = torch.tensor(
            captured_intensity, dtype=torch.float32
        )
        if self.image_domain != 'amp':
            captured_amp = torch.sqrt(captured_intensity)
        else:
            captured_amp = captured_intensity

        return phase_im, captured_amp, angle_rad

    def __add__(self, other):
        """
        Defines addition operation for PairsLoader objects.
        This allows combining two datasets by concatenating their phase_angle_pairs.

        Args:
            other (PairsLoader): Another PairsLoader instance to combine with this one.

        Returns:
            PairsLoader: A new PairsLoader with combined data from both datasets.
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot add PairsLoader with {type(other)}")

        result = self
        result.phase_angle_pairs = self.phase_angle_pairs + other.phase_angle_pairs
        result.length = len(result.phase_angle_pairs)

        # create list of image IDs with augmentation state
        result.order = [(i,) for i in range(len(result.phase_angle_pairs))]
        return result
        return result