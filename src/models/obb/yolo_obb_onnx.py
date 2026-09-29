import numpy as np
from onnxruntime import InferenceSession
from src.models.obb.obb_base import OBBDetectorBase, Model
from src.models.base.yolo_base import ModelError
from src.utils.boxes import xywhr2xyxy, multiclass_nms_class_agnostic_rotated
from src.utils.general import get_classes


class YoloOBBDetector(OBBDetectorBase):
    def __init__(self):
        self._model = None

    def init(self, model_path, class_txt_path, confidence_threshold=0.3, iou_threshold=0.45, is_yolo26=False):
        # is_yolo26 is accepted and stored for parity with the sibling model
        # classes -- both worker threads pass it -- but OBB never branches on
        # it; see inference().
        _class_names = get_classes(class_txt_path)
        _session = InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_names, self.output_names, input_size = self.get_onnx_model_details(_session)
        self._model = Model(
            model=_session,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
            class_names=_class_names,
            is_yolo26=is_yolo26
            )
        init_frame = np.random.randint(0, 256, (input_size[0], input_size[1], 3)).astype(np.uint8)
        self.inference(init_frame)

    def postprocess(self, model_output, scale, conf_threshold, iou_threshold, class_names):
        predictions = np.squeeze(model_output[0]).T
        num_classes = predictions.shape[1] - 5
        scores = np.max(predictions[:, 4:4 + num_classes], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        if predictions.shape[0] == 0:
            return []

        boxes = predictions[:, :4] * scale
        angles = predictions[:, -1]
        dets = multiclass_nms_class_agnostic_rotated(
            boxes, predictions[:, 4:4 + num_classes], angles, iou_threshold, conf_threshold)

        detection_results = []
        if dets is not None:
            xyxy = xywhr2xyxy(np.concatenate([dets[:, :4], dets[:, 6:7]], axis=1))
            for i, det in enumerate(dets):
                obj_dict = {
                        "id": int(i),
                        'class': class_names[int(det[5])],
                        'confidence': det[4],
                        'bbox': np.rint(xyxy[i]),
                        "keypoints": np.array([]),
                        "segmentation": np.array([]),
                        "obb": np.array([det[0], det[1], det[2], det[3], det[6]])}
                detection_results.append(obj_dict)
        return detection_results

    def inference(self, image, confi_thres=None, iou_thres=None):
        if self._model is None:
            raise ModelError("Model not initialized. Have you called init()?")
        if confi_thres is None:
            confi_thres = self._model.confidence_threshold
        if iou_thres is None:
            iou_thres = self._model.iou_threshold

        scale, image = self.preprocess(image, self._model.input_size)

        ort_inputs = {self.input_names[0]: image}
        outputs = self._model.model.run(self.output_names, ort_inputs)

        # OBB is the one task with no NMS-free export. Detection, pose and
        # segmentation yolo26 exports are end-to-end ([1, 300, 6] / [1, 300, 57]
        # / [1, 300, 38]), but yolo26*-obb.onnx emits [1, 4+nc+1, anchors] --
        # the same layout as YOLO11 -- so there is no second postprocess here:
        # is_yolo26 is inert and rotated NMS always runs.
        detection_results = self.postprocess(
            model_output=outputs,
            scale=scale,
            conf_threshold=confi_thres,
            iou_threshold=iou_thres,
            class_names=self._model.class_names
        )
        return detection_results