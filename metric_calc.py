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


import sigopt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


def calc_scores(model, val_data, cfg, label_txt):
    """ Calculate precision, recall, F1-score values on val_data
    """
    X_val, y_val = val_data
    # Get predicted labels
    y_predict = np.asarray(model.predict(X_val))
    y_predict = np.argmax(y_predict, axis=1)
    # Ground truth labels
    y_val = np.argmax(y_val, axis=1)

    # Calculate scores
    val_precision =  precision_score(y_val, y_predict, average='macro')
    val_recall = recall_score(y_val, y_predict, average='macro')
    val_f1score = f1_score(y_val, y_predict, average='macro')

    # Get labels
    f = open(label_txt, "r")
    labels = f.readlines()
    labels = [label[:-1] for label in labels]

    c_matrix = confusion_matrix(y_val, y_predict)

    # Print results
    print("\n\n************** Validation metric scores **************")
    print("\nAverage Precision: ", val_precision)
    print("Average Recall: ", val_recall)
    print("Average F1-score: ", val_f1score)

    print('\nConfusion matrix : ')
    print(labels)
    print(c_matrix)
    print('\n')

    # Update config values with obtained results for logging
    cfg.update({"avg_precision": val_precision})
    cfg.update({"avg_recall": val_recall})
    cfg.update({"avg_f1-score": val_f1score})
    cfg.update({"labels": labels})
    cfg.update({"confusion_matrix": c_matrix.tolist()})

    # Add Sigopt Metrics for optimization
    sigopt.log_metric('precision', val_precision)
    sigopt.log_metric('recall', val_recall)
    sigopt.log_metadata('f1-score', val_f1score)
