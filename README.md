# SSDlite mobilenetv2

This projects use TensorFlow object detection API and MobileNetV2 SSDLite model to train from scatch by using COCO2017 datasets.

## Install

### install tensorflow2 and opencv

    ``` bash
    pip install -r requirements.txt
    ```

### install tensorflow models API

    ``` bash
    git clone https://github.com/tensorflow/models.git 
    cd models/research
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
    ```

## COCO dataset

    The file structure is as follows:<br>
    |--coco<br>
    |--|--annotations<br>
    |--|--images<br>
    |--|--labels<br>
    |--|--tf_record<br>
    |--|--train2017.txt<br>
    |--|--val2017.txt<br>

## How to generate tf_record 

    ``` bash
    python train/create_coco_tf_record_without_test.py
    ```

## Train from scatch

ssdlite_mobilenet_v2_0.5_200x200_coco.config:

reference research/object_detection/samples/configs ssdlite_mobilenet_edgetpu_320x320_coco.config

I set batch 256 ，lr base 0.4，400k num_steps

    ``` bash
    python train_coco.py
    ```

## Saved model

Inference version saved model:

    ``` bash
    python evaluation/export_inference_saved_model.py
    ```

Tflite version saved model:

    ``` bash
    python evaluation/export_tflite_saved_model.py
    ```

## Convert to tflite

    ``` bash
    python evaluation/convert2tflite.py
    ```

## Create txt labels

    ``` bash
    python evaluation/generate_txt_labels.py
    ``` 

## Evaluation

    eval float32 model:

    ``` bash
    python evaluation/eval_coco_tfmodel.py
    ```

    eval int8 model:

    ``` bash
    python evaluation/eval_coco_tflite.py
    ```

## Pretrain model

    ssdlite_mobilenet_v2  aphla=0.5 input_size = 200 x 200:

    checkpoint: coco_model

    tflite: 

## Train on your own dataset
    
    change config file ssdlite_mobilenet_v2_0.5_200x200_coco.config:

    set pretrain model

    change lr and num_steps




