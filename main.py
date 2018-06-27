from keras.models import Model
from keras import applications
from keras import utils
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

import custom_loss as my_loss

BATCH_SIZE = 64
NUM_EPOCHS = 2
MODEL_IMAGE_SIZE = (299, 299)
IMAGE_INTERPOLATION = cv2.INTER_LINEAR
NUM_DEEP_FEATURES = 2048


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    # construct reference model
    ref_model = applications.InceptionV3()
    ref_model.compile(optimizer='sgd', loss='categorical_crossentropy')
    print("Reference model built")

    # construct secondary model with shared layers
    secondary_model = Model(inputs=ref_model.inputs, outputs=ref_model.get_layer('avg_pool').output)
    secondary_model.compile(optimizer='rmsprop', loss=my_loss.doc_total_loss)
    print("Secondary model built")

    # manually train on batches
    ref_training_data_dir = "/home/im-zbox2/harpreet/github/anomaly_data/imagenet_validation"
    ref_training_data_label_file = "/home/im-zbox2/harpreet/github/anomaly_data/ILSVRC2012_validation_ground_truth.txt"

    target_training_data_dir = "/home/im-zbox2/harpreet/github/anomaly_data/train/pos_newsite"

    total_loss_history, discriminative_loss_history = train(ref_model, secondary_model, BATCH_SIZE, NUM_EPOCHS,
                                                            ref_training_data_dir, ref_training_data_label_file,
                                                            target_training_data_dir)
    print("Training completed")

    # save secondary model after training
    save_model_file = "/home/im-zbox2/harpreet/github/anomaly-svm-study/deep_one_class_features/results/trained_model.b.h5"
    secondary_model.save(save_model_file)
    print(f"Model saved to file: {save_model_file}")

    visualize_data(total_loss_history, "total loss history")
    visualize_data(discriminative_loss_history, "discriminative loss history")

    return


def visualize_data(data, title):
    """
    Visualize 2D metrics

    :param data: 2D np array data
    :param title: Title of the graph
    """
    plt.scatter(data[:, 0], data[:, 1], c='blueviolet', s=30, edgecolors='k')
    plt.title(title)
    plt.show()
    return


def train(ref_model, secondary_model, batch_size, num_epochs,
          ref_train_data_dir, ref_train_label_file, target_train_data_dir):
    """
    Manually trains a DOC model

    :param ref_model: Reference model used to maintain discriminative featues
    :param secondary_model: Model used to train on compactness of intra-class features
    :param batch_size: batch size for training
    :param num_epochs: number of epoch to train
    :param ref_train_data_dir: path to ImageNet training data
    :param ref_train_label_file: path to ImageNet training data labels
    :param target_train_data_dir: path to target data
    :return: discriminative loss history and total loss history
    """
    ref_data_labels = read_ref_data_labels(ref_train_label_file)
    ref_images_list = get_images_list(ref_train_data_dir)
    target_images_list = get_images_list(target_train_data_dir)

    num_iterations = max(len(target_images_list) / batch_size, 1) * num_epochs
    print(f"Total number of iterations: {int(num_iterations)}")

    total_loss_history = []
    disc_loss_history = []
    for i in range(int(num_iterations)):
        ref_batch_x, ref_batch_y = read_image_batch(ref_images_list, batch_size, ref_data_labels)
        target_batch_x, target_batch_y = read_image_batch(ref_images_list, batch_size)

        discriminative_loss = ref_model.test_on_batch(ref_batch_x, ref_batch_y)
        disc_loss_history.append((i, discriminative_loss))

        # pass discriminative loss through the placeholder target batch output
        target_batch_y[0][0] = discriminative_loss

        total_loss = secondary_model.train_on_batch(target_batch_x, target_batch_y)
        total_loss_history.append((i, total_loss))

        print(f"Iteration {i}, total loss: {total_loss}, discriminative loss: {discriminative_loss}")

    return np.array(total_loss_history), np.array(disc_loss_history)


def read_image_batch(image_list, batch_size, class_labels=None):
    """
    Read a batch of images

    :param image_list: list of image file names
    :param batch_size: size of batch
    :param class_labels: output labels
    :return: ndarray of images and categorical labels
    """
    batch_images = []
    classification = []
    num_classes = NUM_DEEP_FEATURES if not class_labels else 1000

    for k in range(batch_size):
        rand_loc = random.randrange(0, len(image_list))
        # print(f"opening image: {image_list[rand_loc]}")
        cv_image = read_image(image_list[rand_loc], MODEL_IMAGE_SIZE)

        batch_images.append(cv_image)

        if not class_labels:
            classification.append(0)
        else:
            # print(f"image: {image_list[rand_loc]}, label: {class_labels[rand_loc]}")
            classification.append(class_labels[rand_loc])

    batch_images_np = np.array(batch_images)
    batch_images_np = batch_images_np.astype("float32")
    batch_images_np /= 255.0

    return batch_images_np, utils.to_categorical(np.array(classification), num_classes)


def read_ref_data_labels(data_file):
    """
    Read ImageNet training data labels

    :param data_file:
    :return:
    """
    labels = []
    with open(data_file, 'r') as label_file:
        lines = label_file.readlines()
        # subtracting one to correctly index categorical array
        labels = [int(line) - 1 for line in lines]

    return labels


def get_images_list(list_dir):
    """
    Read all images in a directory

    :param list_dir: path to the root directory
    :return: list of sorted image names
    """
    images_list = []
    for root, sub_dirs, files in os.walk(list_dir):
        images_list += [os.path.join(root, file) for file in files if file.endswith((".jpg", ".JPEG"))]

    images_list.sort()
    return images_list


def read_image(image_file, resize_image=()):
    """
    Read an image and resize it, if necessary

    :param image_file: absolute image path
    :param resize_image: (x, y) tuple for new image dimensions
    :return: cv2 image
    """

    cv_image = cv2.imread(image_file)

    if cv_image is None:
        raise RuntimeError(f"Unable to open {image_file}")

    if len(resize_image) > 0:
        cv_image = cv2.resize(cv_image, resize_image, interpolation=IMAGE_INTERPOLATION)

    return cv_image


if __name__ == '__main__':
    main()
