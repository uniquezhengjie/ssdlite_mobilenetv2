import numpy as np
import tensorflow.compat.v1 as tf
from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import tf_version

def _get_categories_list():
      return [{
      'id': 1,
      'name': 'person'
  }, {
      'id': 2,
      'name': 'dog'
  }, {
      'id': 3,
      'name': 'cat'
  }]

coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
    _get_categories_list())
coco_evaluator.add_single_ground_truth_image_info(
    image_id='image1',
    groundtruth_dict={
        standard_fields.InputDataFields.groundtruth_boxes:
        np.array([[100., 100., 200., 200.]]),
        standard_fields.InputDataFields.groundtruth_classes: np.array([1])
    })
coco_evaluator.add_single_detected_image_info(
    image_id='image1',
    detections_dict={
        standard_fields.DetectionResultFields.detection_boxes:
        np.array([[100., 100., 200., 200.]]),
        standard_fields.DetectionResultFields.detection_scores:
        np.array([.8]),
        standard_fields.DetectionResultFields.detection_classes:
        np.array([1])
    })
coco_evaluator.add_single_ground_truth_image_info(
    image_id='image2',
    groundtruth_dict={
        standard_fields.InputDataFields.groundtruth_boxes:
        np.array([[50., 50., 100., 100.]]),
        standard_fields.InputDataFields.groundtruth_classes: np.array([1])
    })
coco_evaluator.add_single_detected_image_info(
    image_id='image2',
    detections_dict={
        standard_fields.DetectionResultFields.detection_boxes:
        np.array([[50., 50., 100., 100.]]),
        standard_fields.DetectionResultFields.detection_scores:
        np.array([.8]),
        standard_fields.DetectionResultFields.detection_classes:
        np.array([1])
    })
coco_evaluator.add_single_ground_truth_image_info(
    image_id='image3',
    groundtruth_dict={
        standard_fields.InputDataFields.groundtruth_boxes:
        np.array([[25., 25., 50., 50.]]),
        standard_fields.InputDataFields.groundtruth_classes: np.array([1])
    })
coco_evaluator.add_single_detected_image_info(
    image_id='image3',
    detections_dict={
        standard_fields.DetectionResultFields.detection_boxes:
        np.array([[25., 25., 50., 50.]]),
        standard_fields.DetectionResultFields.detection_scores:
        np.array([.8]),
        standard_fields.DetectionResultFields.detection_classes:
        np.array([1])
    })
metrics = coco_evaluator.evaluate()

print(metrics)