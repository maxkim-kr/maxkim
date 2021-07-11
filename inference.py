"""
   Usage
        Single test image:
            python3 tools/inference.py \
                    --model <path/to/OpenVINO/XML/file> \
                    --label ./data/labels.txt \
                    --image ./data/test/<class x>/<img y>.png

        Directory of test images:
            python3 tools/inference.py \
                    --model <path/to/OpenVINO/XML/file> \
                    --label ./data/labels.txt \
                    --dir ./data/test/<class x>/

        Calculate confusion matrix on a test dataset
            python3 tools/inference.py \
                    --model <path/to/OpenVINO/XML/file> \
                    --label ./data/labels.txt \
                    --confusion_matrix \
                    --test_dataset ./data/test/

"""
# Copyright (c) 2020 Intel Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import sys
import cv2
import glob
import time
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.multiclass import unique_labels
from openvino.inference_engine import IENetwork, IECore

SAFE_DIR = str(Path.home())
home_path = sys.path[0]

# Import all library modules
sys.path.append(os.path.dirname(home_path))

from utils.yaml_dict import load_config


def is_safe_path(input_dir):
    """ Check fi input_dir is on allowed safe path
    """
    if not os.path.abspath(input_dir).startswith(SAFE_DIR):
        print("Move directory within safe loc {} to access".format(SAFE_DIR))
        raise argparse.ArgumentTypeError("Directory is not in safe path")
    return input_dir


def parse_inputs():
    """ Parse input arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', '-m', required=True, type=is_safe_path,
                    help='Path to OpenVINO model xml file')
    ap.add_argument('--device', required=False, choices=['CPU', 'GPU', 'HDDL', 'MYRIAD'],
                    help='Specify the target device to infer on : [CPU, GPU, HDDL, MYRIAD]')
    ap.add_argument('--cpu_extension', '-c',
      help='Optional.MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the kernels')
    ap.add_argument('--label', '-l', required=True, type=is_safe_path,
                    help='Path to labels file')
    ap.add_argument('--image', '-i', required=False, type=is_safe_path,
                    help='Path to test image')
    ap.add_argument('--dir', '-d', required=False, type=is_safe_path,
                    help='Path to a dir of images')
    ap.add_argument('--confusion_matrix', action='store_true')
    ap.add_argument('--test_dataset', '-t', required=False, type=is_safe_path,
                    help='Path to test dataset')
    args = ap.parse_args()
    return args


def inference_on_image():
    """ Run inference on a single image
    """
    filename = os.path.basename(args.image)

    img = cv2.imread(args.image)

    if img is None:
        print("\nWARNING : Check input image path. File does not exist\n")
        sys.exit(1)

    # Measure inference time
    start_time = time.time()

    # Preprocess the input image for inferencing
    img_cnn = img.copy()
    img_cnn = cv2.resize(img_cnn, (w, h))
    img_cnn = preprocess_input(img_cnn)
    img_cnn = img_cnn.transpose((2, 0, 1))
    images = np.expand_dims(img_cnn, axis=0)

    # Predict
    res = exec_net.infer(inputs={input_blob: images})
    pred = res[output_blob]
    pred_class = np.argmax(pred)

    end_time = time.time()

    print('')
    print('Test image      : {}'.format(args.image))
    print('Predicted class : {}'.format(class_list[pred_class]))
    print('Inference time  : {} sec'.format(end_time - start_time))
    print('')

    # Display image
    color = (255, 0, 0)
    if (class_list[pred_class] != 'good'):
        color = (0, 0, 255)
    cv2.putText(img, 'Predicted class : {}'.format(class_list[pred_class]),
                (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    cv2.putText(img, 'Inference time  : {:.4f} sec'.format(end_time - start_time),
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    cv2.imshow('Inference result', cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2))))
    cv2.waitKey(0)


def inference_on_dir():
    """ Run inference on a directory of images
    """
    img_types = ('*.png', '*.jpg', '*.jpeg')
    images_l = []
    for t in img_types:
        images_l.extend(glob.glob(args.dir + "/" + t))

    if images_l == []:
        print("\nWARNING : Check directory path. No images to inference.\n")
        sys.exit(1)

    # Count of each defect type
    dtype_count = [0 for i in range(len(class_list))]

    for img_f in images_l:
        img_cnn = cv2.imread(img_f)

        # Measure inference time
        start_time = time.time()

        # Preprocess the input image for inferencing
        img_cnn = cv2.resize(img_cnn, (w, h))
        img_cnn = preprocess_input(img_cnn)
        img_cnn = img_cnn.transpose((2, 0, 1))
        images = np.expand_dims(img_cnn, axis=0)

        # Predict
        res = exec_net.infer(inputs={input_blob: images})
        pred = res[output_blob]
        pred_class = np.argmax(pred)
        dtype_count[pred_class] += 1

        end_time = time.time()

        print('')
        print('Test image      : {}'.format(os.path.basename(img_f)))
        print('Predicted class : {}'.format(class_list[pred_class]))
        print('Inference time  : {} sec'.format(end_time - start_time))
        print('')

    print('\nPrediction results :')
    print('-------------------------')
    for idx, cnt in enumerate(dtype_count):
        print("{0:20} {1}".format(class_list[idx], cnt))
    print('')


def plot_confusion_matrix(true_labels,
                          pred_labels,
                          class_list):
    """ Plot confusion matrix
    """
    # Compute confusion matrix
    matrix = confusion_matrix(true_labels, pred_labels)
    # Get unique label names from the class_list
    classes = class_list[unique_labels(true_labels, pred_labels)]

    print("Confusion matrix : \n")
    print(class_list)
    print(matrix)
    print('\n Multi-class precision: ', precision_score(true_labels, pred_labels, average='macro'))
    print('\n Multi-class recall: ', recall_score(true_labels, pred_labels, average='macro'))
    print('\n Multi-class f1-score: ', f1_score(true_labels, pred_labels, average='macro'))
    print('\n Multi-class accuracy: ', accuracy_score(true_labels, pred_labels, normalize=True))

    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(matrix.shape[1]), classes, rotation=45)
    plt.yticks(np.arange(matrix.shape[0]), classes)

    # Loop over data dimensions and create text annotations
    color_th = 0.5 * matrix.max()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], '.2f'),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if matrix[i, j] > color_th else "black")
    plt.ylabel('Ground truth label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if not os.path.exists('logs'):
        os.mkdir('logs')

    plt.savefig('logs/confusion_matrix.png')
    print('\nConfusion matrix : {}\n'.format('logs/confusion_matrix.png'))
    plt.show()


def generate_confusion_matrix():
    """ Generate confusion matrix
    """
    # NOTE : in this sample code inferencing is done on single images, not as a batch
    # Please refer OpenVINO APIs to perform batch processing

    # Check input test dataset
    if args.test_dataset is None:
        assert False, 'Input test dataset directory'
    # Check if test image dir exists
    if not os.path.exists(args.test_dataset):
        assert False, 'Cannot find test dataset directory'

    img_labels = []
    pred_labels = []

    print('\nLoading data for calculating confusion matrix ...')
    print('* Depending on size of dataset this might take a few minutes *\n')

    # Loop through each class in labels.txt
    for idx, c_name in enumerate(class_list):
        cur_dir = os.path.join(args.test_dataset, c_name)
        images_l = os.listdir(cur_dir)
        print('Processing images in {}'.format(cur_dir))
        # Loop through each image in class
        for img_f in images_l:
            # Load image
            img_cnn = cv2.imread(cur_dir + '/' + img_f)
            # Preprocess the input image for inferencing
            img_cnn = cv2.resize(img_cnn, (w, h))
            img_cnn = preprocess_input(img_cnn)
            img_cnn = img_cnn.transpose((2, 0, 1))
            images = np.expand_dims(img_cnn, axis=0)
            # Predict
            res = exec_net.infer(inputs={input_blob: images})
            pred = res[output_blob]
            pred_class = np.argmax(pred)
            # Append prediction results to pred array
            pred_labels.append(pred_class)
            # Append ground truth label to labels array
            img_labels.append(idx)

    # Plot confusion matrix
    plot_confusion_matrix(img_labels,
                          pred_labels,
                          class_list=np.asarray(class_list))


if __name__ == '__main__':
    """ Main application
    """
    # Parse input arguments
    args = parse_inputs()

    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'

    # --------------------------- Check user inputs ---------------------------
    if not os.path.exists(model_xml):
        raise argparse.ArgumentTypeError("Model XML file {} does not exist"
                                         .format(args.model))
    if not os.path.exists(model_bin):
        raise argparse.ArgumentTypeError("Model BIN file {} does not exist"
                                         .format(model_bin))
    if not os.path.exists(args.label):
        raise argparse.ArgumentTypeError("Label file {} does not exist"
                                         .format(args.label))
    device = 'CPU'
    if args.device:
        if args.device not in ['CPU', 'GPU', 'MYRIAD', 'HDDL']:
            raise argparse.ArgumentTypeError("Invalid device {}"
                                             .format(args.device))
        else:
            device = args.device

    # ---------------------- Load config and labels file ----------------------
    # Load label file
    with open(str(args.label), 'r') as f:
        class_list = sorted([line.strip() for line in f])

    # Load config file
    cfg = load_config(os.path.join(home_path, '../configs/config.yaml'))
    try:
        # Load tensorflow keras applications path
        kerasapp = str("tensorflow.keras.applications")
        # Import module tfkerasapp
        bmodel = __import__(kerasapp, fromlist=[cfg.base_model])
        # Get the base_model call fnction
        app_model = getattr(bmodel, cfg.base_model)
        # Get base_application model funtion from app_model
        basemodel = getattr(app_model, cfg.base_application)
        # Get preprocess_input funtion from app_model
        preprocess_input = getattr(app_model, "preprocess_input")
    except AttributeError:
        print("ERROR : Invalid base model or base application name")
        print("Please check config file")
        sys.exit(1)

    if args.test_dataset is not None and args.confusion_matrix is False:
        print('\nERROR : Incomplete command. Use --confusion_matrix to generate the matrix from test_dataset\n')
        sys.exit(1)

    # ------------------------ OpenVINO initialization ------------------------
    ie = IECore()

    print('Loading network files ...')
    net = ie.read_network(model=model_xml, weights=model_bin)

    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    net.batch_size = 1
    n, c, h, w = net.input_info[input_blob].input_data.shape
    exec_net = ie.load_network(network=net, device_name=device)

    # ------------------------------- Inference -------------------------------
    if args.image:
        inference_on_image()
    if args.dir:
        inference_on_dir()
    if args.confusion_matrix:
        generate_confusion_matrix()
