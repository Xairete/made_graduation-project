import base64
import sys
from io import BytesIO
from typing import Any, List

import torch
import torchvision
from helpers.helpers import ImageMeta
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base_model import BaseModel

sys.path.append("..")

NUM_CLASSES = 2
THRESHOLD_SCORE = 0.6


def postprocess_bbxes(pred: List[Any]):
    bbox_matrices = []
    filtered_pred = []
    for bbox in pred:
        matrix_set = set()
        for x in range(bbox[0], bbox[2]):
            for y in range(bbox[1], bbox[3]):
                matrix_set.add((x, y))
        bbox_matrices.append(matrix_set)

    for i in range(len(bbox_matrices)):
        current_matrix = bbox_matrices[i]
        is_max_contour = False
        for j in range(len(bbox_matrices)):
            compared_matrix = bbox_matrices[j]
            if (len(current_matrix) > len(compared_matrix)) and (len(current_matrix & compared_matrix)/len(compared_matrix) > 0.7):
                is_max_contour = True

        if not is_max_contour:
            filtered_pred.append(pred[i])

    return filtered_pred


class FoodSelector(BaseModel):
    """
    Модель обрезает картинки
    """

    def _init_model(self, model_fname) -> torch.nn.Module:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, NUM_CLASSES)

        for param in model.parameters():
            param.requires_grad = False

        model.load_state_dict(torch.load(
            model_fname, map_location=torch.device('cpu')))
        model.eval()
        return model

    def _get_transform_pipeline(self):
        transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Scale((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transformations

    def __init__(self, model_fname):
        super().__init__(model_fname)

    def predict(self, im_meta: List[ImageMeta]):

        def get_image(image_b: bytes):
            return Image.open(BytesIO(image_b))

        def im_to_bytes(image):
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

        images = [get_image(image.im_bytes) for image in im_meta]

        with torch.no_grad():
            images_tensor = [
                self.img_transform(image).unsqueeze_(0) for image in images
            ]
            images_tensor = torch.cat(images_tensor, 0)
            predictions = self.model(images_tensor)

            images_res = []
            for id_im, pred_im in enumerate(predictions):
                bboxes = self.extract_bboxes(pred_im)
                for bbox in bboxes:
                    # TODO: надо рескейлить и резать нормально
                    crop_image_bytes = im_to_bytes(
                        images[id_im].crop(bbox).resize((256, 256))) 
                    crop_image_b64 = base64.b64encode(
                        crop_image_bytes).decode('ascii')
                    caption = im_meta[id_im].caption
                    images_res.append(
                        ImageMeta(crop_image_bytes, crop_image_b64, caption))

            return images_res

    def extract_bboxes(self, pred_im):
        pred_bboxes = []
        for i in range(len(pred_im['boxes'])):
            x_min, y_min, x_max, y_max = map(
                int, pred_im['boxes'][i].tolist())
            label = int(pred_im['labels'][i].cpu())
            score = float(pred_im['scores'][i].cpu())
            if score > THRESHOLD_SCORE:
                pred_bboxes.append([x_min, y_min, x_max, y_max])
        return postprocess_bbxes(pred_bboxes)
