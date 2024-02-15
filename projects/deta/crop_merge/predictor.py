# inputs      = {"image": image, "height": height, "width": width}
# predictions = self.model([inputs])[0]

import atexit
import bisect
from copy import copy
import multiprocessing as mp
from collections import deque
import cv2
import torch

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = preds.scores > confidence_threshold
        predictions = copy(predictions)  # don't modify the original
        predictions["instances"] = preds[keep_idxs]

    return predictions


class DoubleCropVisualizationDemo(object):
    def __init__(
        self,
        model,
        central_crop_height=300,
        min_image_size=800,
        max_image_size=1333,
        img_format="RGB",
        metadata_dataset="coco_2017_val",
        instance_mode=ColorMode.IMAGE,
    ):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            metadata_dataset if metadata_dataset is not None else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DoubleCropPredictor(model=model,
                                             central_crop_height=central_crop_height,
                                             min_image_size=min_image_size,
                                             max_image_size=max_image_size,
                                             img_format=img_format,
                                             metadata_dataset=metadata_dataset)

    def run_on_image(self, image, threshold=0.5):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        predictions = filter_predictions_with_confidence(
            predictions, threshold)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata,
                                instance_mode=self.instance_mode)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(
                predictions=instances)

        return predictions, vis_output


class DoubleCropPredictor:
    def __init__(
        self,
        model,
        central_crop_height=300,
        min_image_size=800,
        max_image_size=1333,
        img_format="RGB",
        metadata_dataset="coco_2017_val",
    ):
        self.model = model
        # self.model.eval()
        self.metadata = MetadataCatalog.get(metadata_dataset)

        # checkpointer = DetectionCheckpointer(self.model)
        # checkpointer.load(init_checkpoint)

        self.image_aug = T.ResizeShortestEdge(
            [min_image_size, min_image_size], max_image_size)

        self.crop_aug = T.CropTransform(0, int(960 - central_crop_height/2),
                                        3840, central_crop_height)

        self.input_format = img_format
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.image_aug.get_transform(
                original_image).apply_image(original_image)

            print("image shape: ", image.shape)

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            crop = self.crop_aug.apply_image(original_image)
            crop = torch.as_tensor(crop.astype("float32").transpose(2, 0, 1))

            print("image shape: ", image.shape)
            print("crop shape: ", crop.shape)

            inputs = {"image": image, "crop": crop,
                      "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
