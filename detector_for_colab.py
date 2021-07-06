#!/usr/bin/env python
"""
detector.py: This script will take either a video file or capture footage from the first available camera on the
system and pass it through a retinanet model for inference. Finally the frame will be shown along with the bounding
boxes. Press Q to quit the program.
"""
import argparse
import json
import os
from enum import Enum

import cv2
import numpy as np
from noussdk.communication.mappers.mongodb_mapper import LabelToMongo
from openvino.inference_engine import IECore

from utils import (
    draw_boxes_on_image,
    Box,
    aspect_ratio_resize,
    process_outputs,
    DetectionParameters,
    NMSMethod
)

__author__ = "COSMONiO Development Team"
__copyright__ = "COSMONiO © All rights reserved"
__credits__ = ["COSMONiO Development Team"]
__version__ = "1.1"
__maintainer__ = "COSMONiO Development Team"
__email__ = "support@cosmonio.com"
__status__ = "Production"
__updated__ = "12.11.2020"

MODEL_PATHS = {
    "cfg": "model/configurable_parameters.json",
    "weight": "model/inference_model.bin",
    "label_cfg": "model/labels.json"
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class MediaType(Enum):
    IMAGE = 1
    VIDEO = 2


def video_streamer(file_path: str, camera_device: int) -> np.ndarray:
    """
    Make iterator that streams from a video file, or from the specified camera device if no file is specified.

    :param file_path:
    :param camera_device:
    """

    if file_path != '':
        file_path = file_path.replace("\\", "/")
        # capture from video file
        capture = cv2.VideoCapture(file_path)
    else:
        # capture from specified camera device
        capture = cv2.VideoCapture(camera_device)

    while True:
        frame_available, frame = capture.read()
        if not frame_available:
            break

        yield frame
    capture.release()


def image_streamer(file_path: str) -> np.ndarray:
    """
    Make iterator from all images in the specified folder.

    :param file_path:
    """

    file_names = [os.path.join(file_path, file_name) for file_name in os.listdir(file_path)
                  if os.path.splitext(file_name)[-1] in IMAGE_EXTENSIONS]
    for file_name in file_names:
        image = cv2.imread(file_name)
        yield image


def get_configurable_parameters() -> dict:
    """
    Reads configurable parameters from file.
    """
    with open(MODEL_PATHS["cfg"]) as config:
        configurable_parameters = DetectionParameters(json.load(config))
    return configurable_parameters


def get_labels():
    """
    Create labels from the labels.json file stored in the model folder
    """
    with open(MODEL_PATHS["label_cfg"]) as label_file:
        label_data = json.load(label_file)
    labels = [LabelToMongo().backward(label) for label in label_data]
    return labels


def get_shortest_edge(cfg):
    """
    Get the shortest edge of input image
    """
    if cfg.architecture.architecture.value == 'yolov4-tiny':
        return int(cfg.architecture.arch_settings.tiny_image_size.tiny_img_min_side.value)
    return int(cfg.architecture.arch_settings.yolo_image_size.standard_img_min_side.value)


def get_nms_method(cfg) -> NMSMethod:
    """
    Get NMS Method enum
    """
    if cfg.postprocessing.nms_method.value == 'nms':
        return NMSMethod.NMS
    elif cfg.postprocessing.nms_method.value == 'soft-nms':
        return NMSMethod.SOFTNMS


def detector(args):
    """
    Starts a loop that does inference on available frames from passed streamer. If there are no frames available
    the program will stop.
    :param streamer:
    """
    file_name = args.file
    camera_device = args.camera_device

    # create file streamer
    if os.path.isdir(file_name):
        file_streamer = image_streamer(file_name)
        media_type = MediaType.IMAGE
    elif os.path.splitext(file_name)[-1] in IMAGE_EXTENSIONS:
        file_streamer = [cv2.imread(file_name)]
        media_type = MediaType.IMAGE
    else:
        file_streamer = video_streamer(file_name, camera_device)
        media_type = MediaType.VIDEO

    # Read configurable parameters path
    cfg = get_configurable_parameters()
    img_side = get_shortest_edge(cfg)
    nms_method = get_nms_method(cfg)
    conf_thres = float(cfg.postprocessing.confidence_threshold.value)
    nms_thres = float(cfg.postprocessing.iou_threshold.value)
    max_dects = int(cfg.postprocessing.maximum_detection.value)

    # Read label cfg path
    labels = get_labels()

    # Get labels
    num_classes = len(labels)

    model_dir = os.path.dirname(MODEL_PATHS['weight'])
    model_name, model_ext = os.path.splitext(os.path.basename(MODEL_PATHS['weight']))

    assert os.path.exists(model_dir), "OpenVINO model folder does not exist"
    model_xml = os.path.join(model_dir, model_name + ".xml")
    model_bin = os.path.join(model_dir, model_name + ".bin")
    ie = IECore()
    network = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(network.input_info))
    out_blobs = [k for k in network.outputs.keys()]
    network.batch_size = 1
    darknet = ie.load_network(network=network, device_name="CPU")

    for frame in file_streamer:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, C = frame.shape
        input_img = np.full(shape=(img_side, img_side, 3), fill_value=128, dtype=frame.dtype)
        resized_img = aspect_ratio_resize(frame, img_side, img_side)
        img_h, img_w, c = resized_img.shape
        input_img[:img_h, :img_w, :] = resized_img
        input_img = input_img.transpose((2, 0, 1))

        res = darknet.infer(inputs={input_blob: [input_img]})
        preds = res[out_blobs[0]]

        preds = process_outputs(preds, (img_side, img_side), num_classes, conf_thres, nms_thres, max_dects, nms_method)
        pred_dict = preds[0]
        num_pred = pred_dict['num_predictions']
        if num_pred > 0:
            shapes = []
            whwh = np.array([img_w, img_h, img_w, img_h])
            pred_array = np.zeros((num_pred, 6))
            pred_array[:, 0] = pred_dict['pred_classes']
            pred_array[:, 1] = pred_dict['pred_scores']
            pred_array[:, 2:] = pred_dict['pred_boxes']
            for bbox in pred_array:
                bbox[2:] /= whwh
                cls = int(bbox[0])
                score = bbox[1]
                label = labels[cls].name
                color = (labels[cls].color.red, labels[cls].color.green, labels[cls].color.blue)
                shapes.append(
                    Box(x1=bbox[2] * W, y1=bbox[3] * H, x2=bbox[4] * W, y2=bbox[5] * H,
                        labels=[f"{label} {score:.2f}"],
                        color=color)
                )
            canvas = draw_boxes_on_image(frame, shapes)
            cv2.imwrite('car_result.jpg', (canvas))
            wait = 0 if media_type == MediaType.IMAGE else 1  # when dealing with images pause on every image
            if ord("q") == cv2.waitKey(wait):
                break


def main():
    parser = argparse.ArgumentParser("YOLOv4 OpenVINO Detector")
    parser.add_argument("--file",
                        help="Specify a video file location, if nothing is specified it will grab the webcam",
                        type=str,
                        default="")
    parser.add_argument("--camera_device", required=False,
                        help="Specify which camera device to use.",
                        default=0)
    args = parser.parse_args()
    detector(args)


if __name__ == '__main__':
    main()
