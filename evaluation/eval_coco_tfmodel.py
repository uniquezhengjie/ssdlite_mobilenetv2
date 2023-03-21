import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import tf_version

def _get_categories_list(category_index):
      return list(category_index.values())

def detect(detect_fn, input_tensor):
    # input_tensor /= 255.
    # input_tensor -= 0.5
    # input_tensor *= 2.
    input_tensor = tf.round(input_tensor)
    input_tensor = tf.cast(input_tensor, tf.uint8)
    model_fn = detect_fn.signatures['serving_default']
    detections = model_fn(input_tensor)
    boxes = detections['detection_boxes'][0].numpy(),
    classes = detections['detection_classes'][0].numpy().astype(np.int32),
    scores = detections['detection_scores'][0].numpy(),
    num_detections = detections['num_detections'][0].numpy(),
    return boxes[0], classes[0], scores[0], num_detections[0]

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name="detect.png"):
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_annotations,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=True,
    min_score_thresh=0.3)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        return image_np_with_annotations


label_map_path = 'mscoco_label_map.pbtxt'

tf.keras.backend.clear_session()
detect_fn = tf.saved_model.load('coco_saved_model/saved_model')


# Label map can be used to figure out what class ID maps to what
# label. `label_map.txt` is human-readable.
category_index = label_map_util.create_category_index_from_labelmap(
    label_map_path)

label_id_offset = 1
score_thresh = 0.2

coco_evaluator = coco_evaluation.CocoDetectionEvaluator(
    _get_categories_list(category_index))

image_dir = "/data1/datasets/coco/images/val2017/"
label_dir = "/data1/datasets/coco/labels2"

eval_num = 100

for i in os.listdir(label_dir)[:eval_num]:
    name = i.split('.')[0]
    image_path = os.path.join(image_dir, i.replace('txt','jpg'))

    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    image_numpy = image.numpy()
    input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.float32)
    # Note that CenterNet doesn't require any pre-processing except resizing to the
    # input size that the TensorFlow Lite Interpreter was generated with.
    input_tensor = tf.image.resize(input_tensor, (200, 200))
    boxes, classes, scores, num_detections = detect(detect_fn, input_tensor)

    boxes = boxes[scores>score_thresh]
    classes = np.round(classes[scores>score_thresh]).astype(np.uint32)
    scores = scores[scores>score_thresh]
    # print(num_detections)
    # print(classes.shape)
    # vis_image = plot_detections(
    #     image_numpy[0],
    #     boxes,
    #     classes,
    #     scores,
    #     category_index)

    label_path = os.path.join(label_dir, i)
    with open(label_path,'r') as f:
        labels = f.readlines()

    tcls = [c.split()[0] for c in labels]
    tcls = np.array(tcls, dtype=np.uint32)

    tboxes = [c.strip().split()[1:] for c in labels]
    tboxes = np.array(tboxes, dtype=np.float32)
    ntboxes = np.zeros_like(tboxes)
    ntboxes[:,0] = tboxes[:,2]
    ntboxes[:,1] = tboxes[:,0]
    ntboxes[:,2] = tboxes[:,3]
    ntboxes[:,3] = tboxes[:,1]
    # vis_image = plot_detections(
    #     image_numpy[0],
    #     ntboxes,
    #     tcls,
    #     np.ones_like(tcls),
    #     category_index,
    #     image_name="detect_gt.png")

    coco_evaluator.add_single_ground_truth_image_info(
        image_id=name,
        groundtruth_dict={
            standard_fields.InputDataFields.groundtruth_boxes: ntboxes,
            standard_fields.InputDataFields.groundtruth_classes: tcls
        })
    coco_evaluator.add_single_detected_image_info(
        image_id=name,
        detections_dict={
            standard_fields.DetectionResultFields.detection_boxes: boxes,
            standard_fields.DetectionResultFields.detection_scores: scores,
            standard_fields.DetectionResultFields.detection_classes: classes
        })
    # exit()

metrics = coco_evaluator.evaluate()

print(metrics)

