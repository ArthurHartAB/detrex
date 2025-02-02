from detrex.data import DetrDatasetMapper
import copy
import logging
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


class DoubleCropDetrDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into the format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors

    Args:
        augmentation (list[detectron.data.Transforms]): The geometric transforms for
            the input raw image and annotations.
        augmentation_with_crop (list[detectron.data.Transforms]): The geometric transforms with crop.
        is_train (bool): Whether to load train set or val set. Default: True.
        mask_on (bool): Whether to return the mask annotations. Default: False.
        img_format (str): The format of the input raw images. Default: RGB.

    Because detectron2 did not implement `RandomSelect` augmentation. So we provide both `augmentation` and
    `augmentation_with_crop` here and randomly apply one of them to the input raw images.
    """

    def __init__(
        self,
        augmentation,
        augmentation_with_crop,
        is_train=True,
        img_format="RGB"
    ):
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(
                str(self.augmentation), str(self.augmentation_with_crop)
            )
        )

        self.img_format = img_format
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        image = utils.read_image(
            dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        crop = image.copy()

        image, image_transforms = T.apply_transform_gens(
            self.augmentation, image)

        crop, crop_transforms = T.apply_transform_gens(
            self.augmentation_with_crop, crop)

        image_shape = image.shape[:2]  # h, w
        crop_shape = crop.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["crop"] = torch.as_tensor(
            np.ascontiguousarray(crop.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annotations = dataset_dict.pop("annotations")

            image_annos = [
                utils.transform_instance_annotations(
                    obj, image_transforms, image_shape)
                for obj in annotations
                if obj.get("iscrowd", 0) == 0
            ]
            image_instances = utils.annotations_to_instances(
                image_annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(
                image_instances)

            crop_annos = [
                utils.transform_instance_annotations(
                    obj, crop_transforms, crop_shape)
                for obj in annotations
                if obj.get("iscrowd", 0) == 0
            ]
            crop_instances = utils.annotations_to_instances(
                crop_annos, crop_shape)
            dataset_dict["crop_instances"] = utils.filter_empty_instances(
                crop_instances)

        return dataset_dict
