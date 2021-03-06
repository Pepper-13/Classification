#Loading data
from scipy.io import loadmat
import pandas as pd
import numpy as np

class DataLoader(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb', dataset_path=None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'
        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'
        else:
            raise Exception('Incorrect dataset name, please input imdb or fer2013')
            
    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        return ground_truth_data
    
    
    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))
    
    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            faces.append(face)
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions
    
    
#Generating the images
import numpy as np
from random import shuffle
from utils import preprocess_input
from utils import _imread as imread
from utils import _imresize as imresize
from utils import to_categorical
import scipy.ndimage as ndi

class ImageGenerator(object):
    """ Image generator with saturation, brightness, lighting, contrast,
    horizontal flip and vertical flip transformations. It supports
    bounding boxes coordinates.
TODO:
        - Finish support for not using bounding_boxes
            - Random crop
            - Test other transformations
    """
    def __init__(self, ground_truth_data, batch_size, image_size,
                train_keys, validation_keys,
                ground_truth_transformer=None,
                path_prefix=None,
                saturation_var=0.5,
                brightness_var=0.5,
                contrast_var=0.5,
                lighting_std=0.5,
                horizontal_flip_probability=0.5,
                vertical_flip_probability=0.5,
                do_random_crop=False,
                zoom_range=[0.75, 1.25],
                translation_factor=.3):
        self.ground_truth_data = ground_truth_data
        self.ground_truth_transformer = ground_truth_transformer
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.validation_keys = validation_keys
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_random_crop = do_random_crop
        self.zoom_range = zoom_range
        self.translation_factor = translation_factor
        
    def _do_random_crop(self, image_array):
        """IMPORTANT: random crop only works for classification since the
        current implementation does no transform bounding boxes"""
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                        self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                        crop_matrix, offset=offset, order=0, mode='nearest',
                        cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array
    
    def do_random_rotation(self, image_array):
        """IMPORTANT: random rotation only works for classification since the
        current implementation does no transform bounding boxes"""
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                        self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                        crop_matrix, offset=offset, order=0, mode='nearest',
                        cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array
    
    
    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array + (1 - alpha) * gray_scale[:, :, None]
        return np.clip(image_array, 0, 255)
    
    def brightness(self, image_array):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)
    
     def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                        np.ones_like(image_array))
        alpha = 2 * np.random.random() * self.contrast_var
        alpha = alpha + 1 - self.contrast_var
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)
    
    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1,3) /
                                    255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array = image_array + noise
        return np.clip(image_array, 0 ,255)
    
    
    def horizontal_flip(self, image_array, box_corners=None):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            if box_corners != None:
                box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
        return image_array, box_corners

    def vertical_flip(self, image_array, box_corners=None):
        if (np.random.random() < self.vertical_flip_probability):
            image_array = image_array[::-1]
            if box_corners != None:
                box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
        return image_array, box_corners
    
    def transform(self, image_array, box_corners=None):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array = jitter(image_array)

        if self.lighting_std:
            image_array = self.lighting(image_array)

        if self.horizontal_flip_probability > 0:
            image_array, box_corners = self.horizontal_flip(image_array,
                                                            box_corners)

        if self.vertical_flip_probability > 0:
            image_array, box_corners = self.vertical_flip(image_array,
                                                            box_corners)
            
        return image_array, box_corners

    def preprocess_images(self, image_array):
        return preprocess_input(image_array)
    
    def flow(self, mode='train'):
            while True:
                if mode =='train':
                    shuffle(self.train_keys)
                    keys = self.train_keys
                elif mode == 'val' or  mode == 'demo':
                    shuffle(self.validation_keys)
                    keys = self.validation_keys
                else:
                    raise Exception('invalid mode: %s' % mode)

                inputs = []
                targets = []
                for key in keys:
                    image_path = self.path_prefix + key
                    image_array = imread(image_path)
                    image_array = imresize(image_array, self.image_size)

                    num_image_channels = len(image_array.shape)
                    if num_image_channels != 3:
                        continue

                    ground_truth = self.ground_truth_data[key]

                    if self.do_random_crop:
                        image_array = self._do_random_crop(image_array)

                    image_array = image_array.astype('float32')
                    if mode == 'train' or mode == 'demo':
                        if self.ground_truth_transformer != None:
                            image_array, ground_truth = self.transform(
                                                                image_array,
                                                                ground_truth)
                            ground_truth = (
                                self.ground_truth_transformer.assign_boxes(
                                                            ground_truth))
                        else:
                            image_array = self.transform(image_array)[0]

                    inputs.append(image_array)
                    targets.append(ground_truth)
                    if len(targets) == self.batch_size:
                        inputs = np.asarray(inputs)
                        targets = np.asarray(targets)
                        # this will not work for boxes
                        targets = to_categorical(targets)
                        if mode == 'train' or mode == 'val':
                            inputs = self.preprocess_images(inputs)
                            yield self._wrap_in_dictionary(inputs, targets)
                        if mode == 'demo':
                            yield self._wrap_in_dictionary(inputs, targets)
                        inputs = []
                        targets = []
                        
    def _wrap_in_dictionary(self, image_array, targets):
        return [{'image_array_input':image_array},
                {'predictions':targets}]
    
    
    #Models
    
    from keras.layers import Activation, Convolution2D, Dropout
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential

def simple_CNN(input_shape, num_classes):

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax',name='predictions'))
    return model

#train emotional classifier

from keras.callbacks import CSVLogger, ModelCheckpoint
from data_loader import DataLoader
from models import simple_CNN
from utils import preprocess_input

# parameters
batch_size = 128
num_epochs = 1000
training_split = .8
dataset_name = 'fer2013'
log_file_path = 'log_files/emotion_training.log'
trained_models_path = '../trained_models/emotion_models/simple_CNN'

# data loader
data_loader = DataLoader(dataset_name)
faces, emotions = data_loader.get_data()
print(len(faces))
faces = preprocess_input(faces)
num_classes = emotions.shape[1]
input_shape = faces.shape[1:]

# model parameters/compilation
model = simple_CNN(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                                        metrics=['accuracy'])
model.summary()

# model callbacks
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                            'val_acc', verbose=1,
                                save_best_only=True)
callbacks = [model_checkpoint, csv_logger]

# model training
model.fit(faces, emotions, batch_size, num_epochs,verbose=1,
                                        callbacks=callbacks,
                        validation_split=(1-training_split),
                                                shuffle=True)  

#train_gender_classifier

from keras.callbacks import CSVLogger, ModelCheckpoint
from data_loader import DataLoader
from models import simple_CNN
from image_generator import ImageGenerator
from utils import split_data

# parameters
batch_size = 32
num_epochs = 1000
training_split = .8
do_random_crop = False
num_classes = 2
dataset_name = 'imdb'
input_shape = (48, 48, 3)
images_path = '../datasets/imdb_crop/'
log_file_path = 'log_files/gender_training.log'
trained_models_path = '../trained_models/gender_models/simple_CNN'

# data loader
data_loader = DataLoader(dataset_name)
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_data(ground_truth_data, training_split)
image_generator = ImageGenerator(ground_truth_data, batch_size,
                                input_shape[:2],
                                train_keys, val_keys, None,
                                path_prefix=images_path,
                                vertical_flip_probability=0,
                                do_random_crop=do_random_crop)

# model parameters/compilation
model = simple_CNN(input_shape, num_classes)
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

# model callbacks
csv_logger = CSVLogger(log_file_path, append=False)
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False)

# model training
model.fit_generator(image_generator.flow(mode='train'),
                    steps_per_epoch=int(len(train_keys) / batch_size),
                    epochs=num_epochs, verbose=1,
                    callbacks=[csv_logger, model_checkpoint],
                    validation_data= image_generator.flow('val'),
                    validation_steps=int(len(val_keys) / batch_size))

#Adding Images for testing

import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from utils import preprocess_input
from utils import get_labels

# parameters
image_path = '../images/test_image.jpg'
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/simple_CNN.530-0.65.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

x_offset_emotion = 20
y_offset_emotion = 40
x_offset = 30
y_offset = 60

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
gender_classifier = load_model(gender_model_path)

frame = cv2.imread(image_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
faces = face_detection.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    face = frame[(y - y_offset):(y + h + y_offset),
                (x - x_offset):(x + w + x_offset)]

    gray_face = gray[(y - y_offset_emotion):(y + h + y_offset_emotion),
                    (x - x_offset_emotion):(x + w + x_offset_emotion)]
    try:
        face = cv2.resize(face, (48, 48))
        gray_face = cv2.resize(gray_face, (48, 48))
    except:
        continue
    face = np.expand_dims(face, 0)
    face = preprocess_input(face)
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]

    gray_face = preprocess_input(gray_face)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    
    if gender == gender_labels[0]:
        gender_color = (0, 0, 255)
    else:
        gender_color = (255, 0, 0)

    cv2.rectangle(frame, (x, y), (x + w, y + h), gender_color, 2)
    cv2.putText(frame, emotion, (x, y - 90), font,
                    2, gender_color, 2, cv2.LINE_AA)
    cv2.putText(frame, gender, (x , y - 90 + 70), font,
                    2, gender_color, 2, cv2.LINE_AA)
    
    
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/predicted_test_image.png', frame)

#Video Test
import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from utils import preprocess_input
from utils import get_labels

# parameters
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/simple_CNN.530-0.65.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
frame_window = 10
x_offset_emotion = 20
y_offset_emotion = 40
x_offset = 30
y_offset = 60

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
gender_classifier = load_model(gender_model_path)

# video 
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')
emotion_label_window = []
gender_label_window = []
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        face = frame[(y - y_offset):(y + h + y_offset),
                    (x - x_offset):(x + w + x_offset)]

        gray_face = gray[(y - y_offset_emotion):(y + h + y_offset_emotion),
                        (x - x_offset_emotion):(x + w + x_offset_emotion)]
        try:
            face = cv2.resize(face, (48, 48))
            gray_face = cv2.resize(gray_face, (48, 48))
        except:
            continue
        face = np.expand_dims(face, 0)
        face = preprocess_input(face)
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        gender_label_window.append(gender)

        gray_face = preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]
        emotion_label_window.append(emotion)

        if len(gender_label_window) >= frame_window:
            emotion_label_window.pop(0)
            gender_label_window.pop(0)
        try:
            emotion_mode = mode(emotion_label_window)
            gender_mode = mode(gender_label_window)
        except:
            continue
        if gender_mode == gender_labels[0]:
            gender_color = (255, 0, 0)
        else:
            gender_color = (0, 255, 0)

        #cv2.rectangle(frame, (x - x_offset, y - y_offset),
                    #(x + w + x_offset, y + h + y_offset),
        cv2.rectangle(frame, (x, y), (x + w, y + h), gender_color, 2)
        cv2.putText(frame, emotion_mode, (x, y - 30), font,
                        .7, gender_color, 1, cv2.LINE_AA)
        cv2.putText(frame, gender_mode, (x + 90, y - 30), font,
                        .7, gender_color, 1, cv2.LINE_AA)
        
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', frame)
    except:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

    


    
