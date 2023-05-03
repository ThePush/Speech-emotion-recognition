import os
import sys
import numpy as np
from keras.models import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


def main():
    print('*' * 80)
    if len(sys.argv) != 2:
        print('Usage: python stats.py <dataset_mode>')
        print('Example: python stats.py RAVDESS_sp_14')
        sys.exit(1)

    MODE = sys.argv[1]
    MODE_VALID = MODE
    MODEL_PATH = 'models/model_' + MODE + '.ckpt'

    labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']

    x_test = np.load('np_arrays/' + MODE_VALID +
                     '/x_' + MODE_VALID + '.npy')
    y_test = np.load('np_arrays/' + MODE_VALID +
                     '/y_' + MODE_VALID + '.npy')
    print('y_test: ', y_test)

    model = load_model(MODEL_PATH)
    y_pred = model.predict(x_test).argmax(axis=1)
    print(y_test.shape, y_pred.shape)
    print('y_pred: ', y_pred)

    print('Classification report: ')
    print(classification_report(y_test, y_pred,
          target_names=labels, labels=range(len(labels))))
    print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')

    # plt.xkcd()
    cm = confusion_matrix(y_test, y_pred, labels=range(len(labels)))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    display.plot(cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    # plt.savefig('figures/' + MODE_VALID + '/' + MODE_VALID + '_confusion_matrix_test.png')
    plt.show()


if __name__ == '__main__':
    main()
