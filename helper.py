
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
import os

def explore_directory(directory_path):
    """
    Explores the directory and returns a list of all the files in the directory.

    Args:
        directory_path (str): target directory

    """
    for dirpath, dirnames, filenames in os.walk(directory_path):
        print(f"Found {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_acc_loss_curves(history):
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
    
def show_confusion_matrix(y_true, y_pred):
    """
    Generates a confusion matrix for the model.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.
    """
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
    plt.show()
    
def multiclass_roc_auc_score(y_true, y_pred, labels, average='macro'):
    """
    Generates a ROC curve for the model.

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.
        labels (list): list of labels.
        average (str, optional): average method for the ROC curve. Defaults to 'macro'.

    Returns:
        float: ROC AUC score.
    """
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        ax.plot(fpr, tpr, label= '%s (AUC: %0.2f)' % (label, auc(fpr, tpr)))
    ax.plot(fpr, tpr, 'b-', label='Random Guessing')
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    return roc_auc_score(y_true, y_pred, average=average)

def view_random_image(target_dir, target_cls):
    """
    View a random image from the target directory.
    
    Args:
        target_dir (str): target directory.
        target_cls (str): target class.
    """
    target_folder = os.path.join(target_dir, target_cls)
    random_image = np.random.choice(os.listdir(target_folder))
    image = plt.imread(os.path.join(target_folder, random_image))
    plt.imshow(image)
    plt.title(f"{target_cls}\n({random_image.shape})")
    plt.axis('off')
    
    