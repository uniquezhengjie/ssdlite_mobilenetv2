# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw COCO dataset to TFRecord for object_detection.

This tool supports data generation for object detection (boxes, masks),
keypoint detection, and DensePose.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os
import contextlib2
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_string('val_annotations_file', '/data1/datasets/coco/annotations/instances_val2017.json',
                       'Validation annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/data1/datasets/coco/labels2', 'Output data directory.')


FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_COCO_KEYPOINT_NAMES = [
    b'nose', b'left_eye', b'right_eye', b'left_ear', b'right_ear',
    b'left_shoulder', b'right_shoulder', b'left_elbow', b'right_elbow',
    b'left_wrist', b'right_wrist', b'left_hip', b'right_hip',
    b'left_knee', b'right_knee', b'left_ankle', b'right_ankle'
]

_COCO_PART_NAMES = [
    b'torso_back', b'torso_front', b'right_hand', b'left_hand', b'left_foot',
    b'right_foot', b'right_upper_leg_back', b'left_upper_leg_back',
    b'right_upper_leg_front', b'left_upper_leg_front', b'right_lower_leg_back',
    b'left_lower_leg_back', b'right_lower_leg_front', b'left_lower_leg_front',
    b'left_upper_arm_back', b'right_upper_arm_back', b'left_upper_arm_front',
    b'right_upper_arm_front', b'left_lower_arm_back', b'right_lower_arm_back',
    b'left_lower_arm_front', b'right_lower_arm_front', b'right_face',
    b'left_face',
]

_DP_PART_ID_OFFSET = 1


def clip_to_unit(x):
  return min(max(x, 0.0), 1.0)


def create_tf_example(image,
                      annotations_list,
                      output_path,
                      category_index):

  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  encoded_mask_png = []
  keypoints_x = []
  keypoints_y = []
  keypoints_visibility = []
  keypoints_name = []
  num_keypoints = []
  num_annotations_skipped = 0
  num_keypoint_annotation_used = 0
  num_keypoint_annotation_skipped = 0
  dp_part_index = []
  dp_x = []
  dp_y = []
  dp_u = []
  dp_v = []
  dp_num_points = []
  densepose_keys = ['dp_I', 'dp_U', 'dp_V', 'dp_x', 'dp_y', 'bbox']
  num_densepose_annotation_used = 0
  num_densepose_annotation_skipped = 0
  for object_annotations in annotations_list:
    (x, y, width, height) = tuple(object_annotations['bbox'])
    if width <= 0 or height <= 0:
      num_annotations_skipped += 1
      continue
    if x + width > image_width or y + height > image_height:
      num_annotations_skipped += 1
      continue
    category_id = int(object_annotations['category_id'])
    # print(category_id)
    category_name = category_index[category_id]['name'].encode('utf8')

    xmin.append(float(x) / image_width)
    xmax.append(float(x + width) / image_width)
    ymin.append(float(y) / image_height)
    ymax.append(float(y + height) / image_height)
    is_crowd.append(object_annotations['iscrowd'])
    category_ids.append(category_id)
    category_names.append(category_name)
    area.append(object_annotations['area'])

  for idx,cid in enumerate(category_ids):
    with open(os.path.join(output_path,filename.replace(".jpg",".txt")),"a+") as f:
      cont = str(cid) + " " + str(xmin[idx]) + " " + str(xmax[idx]) + " " +str(ymin[idx]) + " " + str(ymax[idx])+ "\n"
      f.write(cont)

  return num_annotations_skipped


def _create_tf_record_from_coco_annotations(annotations_file, output_path, num_shards):

  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    groundtruth_data = json.load(fid)
    images = groundtruth_data['images']
    category_index = label_map_util.create_category_index(
        groundtruth_data['categories'])
    # print(category_index)
    annotations_index = {}
    if 'annotations' in groundtruth_data:
      logging.info('Found groundtruth annotations. Building annotations index.')
      for annotation in groundtruth_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_index:
          annotations_index[image_id] = []
        annotations_index[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
      image_id = image['id']
      if image_id not in annotations_index:
        missing_annotation_count += 1
        annotations_index[image_id] = []
    logging.info('%d images are missing annotations.',
                 missing_annotation_count)

    total_num_annotations_skipped = 0
    for idx, image in enumerate(images):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(images))
      annotations_list = annotations_index[image['id']]

      num_annotations_skipped= create_tf_example(
           image, annotations_list, output_path, category_index)
      total_num_annotations_skipped += num_annotations_skipped
      shard_idx = idx % num_shards
    logging.info('Finished writing, skipped %d annotations.',
                 total_num_annotations_skipped)


def main(_):

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  _create_tf_record_from_coco_annotations(
      FLAGS.val_annotations_file,
      FLAGS.output_dir,num_shards=50)


if __name__ == '__main__':
  tf.app.run()