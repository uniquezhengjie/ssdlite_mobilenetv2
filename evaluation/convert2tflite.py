import tensorflow as tf
import numpy as np
from pathlib import Path
import glob
import os
import cv2
from PIL import Image
import random

random.seed = 42


def representative_dataset_gen(dataset, ncalib=100):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    for n, (path, img) in enumerate(dataset):
        img = img[..., ::-1]
        x = np.expand_dims(img, axis=0).astype(np.float32)
        x /= 255.
        x -= 0.5
        x *= 2.
        yield [x]
        if n >= ncalib:
            break


class LoadImages:
    #  image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=(640, 640), stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        random.shuffle(self.files)
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        self.count += 1
        im = Image.open(path)
        im = im.convert('RGB')
        im = im.resize(self.img_size, Image.BILINEAR)
        im = np.array(im)

        # Convert
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(im)

        return path, img

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


model_path = 'coco_model_tflite/saved_model'
# model_path = 'model3_export/saved_model/' # 使用export.py导出的savedmodel 无法生成tflite

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
dataset = LoadImages("/data1/datasets/coco/images/val2017", img_size=(200, 200), auto=False)
converter.representative_dataset = lambda: representative_dataset_gen(dataset, 100)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = []
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# converter.experimental_new_quantizer = False
tflite_model = converter.convert()
open('ssdlite_mobilenet_v2_coco_0.5_200x200.tflite', "wb").write(tflite_model)
