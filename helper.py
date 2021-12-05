import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import os

def set_seeds(seed):
    """
    Sets the seeds for the following: PYTHONHASHSEED, NumPy, and TensorFlow.

    Args:
        `seed` (int): seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def explore_directory(directory_path):
    """
    Explores the directory and returns a list of all the files in the directory.

    Args:
        `directory_path` (str): target directory
    """
    for dirpath, dirnames, filenames in os.walk(directory_path):
        print(f"Found {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_acc_loss_curves(history):
    """
    Returns separate loss and accuracy curves from the history object for the training and validation sets.

    Args:
        `history` (keras.callbacks.History): history object returned by the fit method of the model.
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
        `y_true` (list): list of true labels.
        `y_pred` (list): list of predicted labels.
    """
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap="YlGnBu")
    plt.show()
    
def multiclass_roc_auc_score(y_true, y_pred, labels, average='macro'):
    """
    Generates a ROC curve for the model.

    Args:
        `y_true` (list): list of true labels.
        `y_pred` (list): list of predicted labels.
        `labels` (list): list of labels.
        `average` (str, optional): average method for the ROC curve. Defaults to 'macro'.

    Returns:
        float: ROC AUC score.
    """
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    
    _, ax = plt.subplots(figsize=(10, 10))
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
        `target_dir` (str): target directory.
        `target_cls` (str): target class.
    """
    target_folder = os.path.join(target_dir, target_cls)
    random_image = np.random.choice(os.listdir(target_folder))
    image = plt.imread(os.path.join(target_folder, random_image))
    plt.imshow(image)
    plt.title(f"{target_cls}\n({random_image.shape})")
    plt.axis('off')

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15): 
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
        `y_true` (list): list of true labels.
        `y_pred` (list): list of predicted labels.
        `classes` (list, optional): list of classes. Defaults to None.
        `figsize` (tuple, optional): figure size. Defaults to (10, 10).
        `text_size` (int, optional): text size. Defaults to 15.
        
    Returns:
        matplotlib.pyplot.figure: figure object.
    
    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    """  
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
    
def create_model(model_url, image_shape=(224,224), num_classes=10):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
    
    Args:
        `model_url` (str): TensorFlow Hub feature extraction URL.
        `image_shape` (tuple, optional): image shape. Defaults to (224,224).
        `num_classes` (int, optional): number of classes. Defaults to 10.
    
    Returns:
        keras.models.Sequential: Keras Sequential model with model_url as feature
        extractor layer and Dense output layer with num_classes outputs.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                            trainable=False, # freeze the underlying patterns
                                            name='feature_extraction_layer',
                                            input_shape=image_shape+(3,)) # define the input image shape
    
    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer, # use the feature extraction layer as the base
        layers.Dense(num_classes, activation='softmax', name='output_layer') # create our own output layer      
    ])

    return model
    