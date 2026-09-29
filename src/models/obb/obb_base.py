from collections import namedtuple
from src.models.base.yolo_base import YoloPredictorBase
from src.utils.visualize import PALLETE, draw_obb, draw_label, label_metrics


Model = namedtuple("Model", "model confidence_threshold iou_threshold input_size class_names is_yolo26")


class OBBDetectorBase(YoloPredictorBase):
    @staticmethod
    def draw_results(image, model_results):
        img_cpy = image.copy()
        if model_results == []:
            return img_cpy
        height, width, _ = img_cpy.shape
        font_scale, thickness = label_metrics(width, height)
        for obj in model_results:
            id = int(obj["id"])
            color = PALLETE[id % PALLETE.shape[0]]
            text = '%d-%s' % (id, obj["class"])
            anchor = draw_obb(img_cpy, obj["obb"], color, int(thickness * 5 * font_scale))
            draw_label(img_cpy, text, anchor, color, font_scale, thickness)
        return img_cpy