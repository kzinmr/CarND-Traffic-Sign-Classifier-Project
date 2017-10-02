import csv
import os
import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

NUM_CLASSES=43
IMG_SIZE = 32
NORMALIZE = False
GRAYSCALE = False
NUM_CHANNELS = 1 if GRAYSCALE else 3


def get_image_paths(rootpath='./GTSRB/Training'):
    image_paths = []
    labels = []
    for c in range(0, NUM_CLASSES):
        prefix = os.path.join(rootpath, format(c, '05d'))
        with open(os.path.join(prefix, 'GT-{0:05d}.csv'.format(c))) as fp:
            annotation_csv = csv.reader(fp, delimiter=';')
            header = next(annotation_csv)
            for row in annotation_csv:
                # width, height = row[1], row[2]
                image_paths.append(os.path.join(prefix, row[0]))
                labels.append(int(row[7]))
    return image_paths, labels


def rgb_to_gray(images):
    images_gray = np.average(images, axis=3)
    # shape: (batch_size, 32, 32)
    images_gray = np.expand_dims(images_gray, axis=3)
    # shape: (batch_size, 32, 32, 1)
    return images_gray


def normalize_pixels(X):
    # convert uint8 images: [0, 255] -> [-1, 1]
    X = X.astype('float32')
    X = (X - 128.) / 128.
    return X


def preprocess_images(X, grayscale=False, normalize=False):
    if grayscale:
        X = rgb_to_gray(X)
    if normalize:
        X = normalize_pixels(X)
    return X


def read_images_from_paths(image_paths):
    # Read image files, resize them, convert to numpy arrays w/ dtype=uint8
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        image = np.array(list(image.getdata()), dtype='uint8')
        image = np.reshape(image, (IMG_SIZE, IMG_SIZE, 3))
        images.append(image)
    images = np.array(images, dtype='uint8')
    return images


def read_traffic_images(rootpath):
    images_paths, labels = get_image_paths(rootpath)
    images = read_images_from_paths(images_paths) # uint8
    images = preprocess_images(images, GRAYSCALE, NORMALIZE) # float if NORMALIZE
    return images, labels


def onehot_encoding(y, n_classes):
    ohe_labels = np.zeros((y.shape[0], n_classes))
    for i, ohe_label in enumerate(ohe_labels):
        ohe_labels[y[i]] = 1.
    return ohe_labels


def load_signnames(root_path='.'):
    # Load signnames.csv to map label number to sign string
    label2signname = {}
    with open(root_path+'signnames.csv', 'r') as fp:
        signnames_csv = csv.reader(fp, delimiter=',')
        header = annotation_csv.next()
        for label, signname in signnames_csv:
            label_map[int(label)] = signname
    return label2signname


if __name__ == '__main__':
    rp_img = './GTSRB/Final_Training/Images'
    images, labels = read_traffic_images(rp_img)
    rp_data = './data'
    with open(os.path.join(rp_data, 'images.pkl'), 'wb') as fp:
        pickle.dump(images, fp)
    with open(os.path.join(rp_data, 'labels.pkl'), 'wb') as fp:
        pickle.dump(labels, fp)
    X, y = images, labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('train:{}, test:{}'.format(len(y_train), len(y_test)))
    with open(os.path.join(rp_data, 'train.pkl'), 'wb') as fp:
        train = {'features': X_train, 'labels': y_train}
        pickle.dump(train, fp)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    print('valid:{}, test:{}'.format(len(y_valid), len(y_test)))
    with open(os.path.join(rp_data, 'validation.pkl'), 'wb') as fp:
        valid = {'features': X_valid, 'labels': y_valid}
        pickle.dump(valid, fp)
    with open(os.path.join(rp_data, 'test.pkl'), 'wb') as fp:
        test = {'features': X_test, 'labels': y_test}
        pickle.dump(test, fp)
