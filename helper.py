
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

def explore_directory(directory_path: str):
    """
    Explores the directory and returns a list of all the files in the directory.

    Args:
        directory_path (str): target directory

    """
    for dirpath, dirnames, filenames in os.walk(directory_path):
        print(f"Found {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_acc_loss_curves(history: dict):
    """
    Returns separate loss and accuracy curves from the history object for the training and validation sets.

    Args:
        history (keras.callbacks.History): history object returned by the fit method of the model.
    """
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(history.history['loss']))

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')

    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy Loss')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.show()
    
def show_classification_report(model, images, labels):
    """
    Generates a classification report for the model.

    Args:
        model (keras.models.Model): model to be evaluated.
        images (list): list of images to be classified.
        labels (list): list of labels.
    """
    
    y_true = images.classes
    y_pred = np.argmax(model.predict(images), axis=1)
    print(classification_report(y_true, y_pred, target_names=labels))
    
    
