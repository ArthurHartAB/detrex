import torch.nn as nn
import torch

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

IMAGE_HEIGHT = 1920


class TwoCropsWrapper(nn.Module):
    def __init__(self, model,
                 crop_height=300,
                 crop_pos=IMAGE_HEIGHT/2,
                 border_delta=10,
                 intersection_delta=20):
        super(TwoCropsWrapper, self).__init__()
        self.model = model
        self.crop_height = crop_height
        self.crop_pos = crop_pos
        self.border_delta = border_delta
        self.intersection_delta = intersection_delta

    def forward(self, inputs):

        predictions_image = self.model(inputs)
        predictions_crop = self.model(
            [{"image": input["crop"], "height": input["height"], "width": input["width"]} for input in inputs])

        merged_predictions = self.merge_predictions(
            predictions_image, predictions_crop)

        return merged_predictions

    def merge_predictions(self, predictions_image, predictions_crop):
        merged_predictions = [self.merge_single_predictions(
            predictions_image[i], predictions_crop[i]) for i in range(len(predictions_crop))]
        return merged_predictions

    def merge_single_predictions(self, predictions_single_image, predictions_single_crop):
        # merge the predictions

        image_boxes = predictions_single_image['instances'].pred_boxes.tensor

        crop_boxes = predictions_single_crop['instances'].pred_boxes.tensor

        print("crop_boxes : ", crop_boxes)

        crop_boxes[:, 1] = crop_boxes[:, 1] * \
            self.crop_height/IMAGE_HEIGHT + IMAGE_HEIGHT/2 - self.crop_height/2
        crop_boxes[:, 3] = crop_boxes[:, 3] * \
            self.crop_height/IMAGE_HEIGHT + IMAGE_HEIGHT/2 - self.crop_height/2

        crop_det_bottom = crop_boxes[:, 3]
        crop_det_top = crop_boxes[:, 1]

        boundary_bottom = self.crop_pos + \
            (self.crop_height / 2) - self.border_delta
        boundary_top = self.crop_pos - \
            (self.crop_height / 2) + self.border_delta

        is_crop_det_in_boundary = (crop_det_bottom < boundary_bottom) & (
            crop_det_top > boundary_top)

        predictions_single_crop['instances'] = predictions_single_crop['instances'][is_crop_det_in_boundary]

        image_det_bottom = image_boxes[:, 3]
        image_det_top = image_boxes[:, 1]

        is_image_det_outside_boundary = (image_det_bottom > boundary_bottom - self.intersection_delta) | (
            image_det_top < boundary_top + self.intersection_delta)

        predictions_single_image['instances'] = predictions_single_image['instances'][is_image_det_outside_boundary]

        merged_instances = Instances.cat([predictions_single_image['instances'],
                                          predictions_single_crop['instances']])

        return {"instances": merged_instances}
