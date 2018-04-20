from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras.utils import plot_model
import numpy as np
import csv
import h5py

# file_name = 'middle_feature.h5'
# model_file = 'model.h5'

# is_load_weight = False

def extract_feature(train_path, test_path, file_name, image_size=(299, 299)):

    img_gen = ImageDataGenerator(rescale=1./255)
    train_generator = img_gen.flow_from_directory(
            train_path,
            target_size=image_size,
            class_mode='categorical',
            classes=['AMD','DME','NOR'],
            shuffle=False)
    test_generator = img_gen.flow_from_directory(
            test_path,
            target_size=image_size,
            class_mode='categorical',
            classes=['AMD','DME','NOR'],
            shuffle=False)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed8').output)

    train = model.predict_generator(train_generator)
    test = model.predict_generator(test_generator)

    f = h5py.File(file_name, 'w')
    f.create_dataset("train", data=train)
    f.create_dataset("test", data=test)
    f.create_dataset("train_label", data=train_generator.classes)
    f.create_dataset("test_label", data=test_generator.classes)
    f.close()

def build_model(file_name, model_file, kernel_num = 128):

    f = h5py.File(file_name, 'r')
    X_train = np.array(f['train'])
    Y_train = np.array(f['train_label'])
    f.close()

    Y_train = np_utils.to_categorical(Y_train)

    input_tensor = Input(X_train.shape[1:])
    x = input_tensor
    x = Conv2D(kernel_num, 3, padding='same')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(kernel_num, 3, padding='same')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(kernel_num, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=predictions)
    model.compile(optimizer='Adadelta', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    
    for i in range(50):
        print ("round: " + str(i + 1))
        model.fit(X_train, Y_train)
        train_acc = model.evaluate(X_train, Y_train)
        print ('train_acc = ' + str(train_acc[1]))
    
    model.save(model_file)

def model_predict(file_name, model_file, result_path):
    f = h5py.File(file_name, 'r')
    X_test = np.array(f['test'])
    Y_test = np.array(f['test_label'])
    f.close()

    Y_test = np_utils.to_categorical(Y_test)
    model = load_model(model_file)

    test_acc = model.evaluate(X_test, Y_test)
    print ('Test Accuarcy: ' + str(test_acc[1]))

    csvfile = open(result_path, "w")
    writer = csv.writer(csvfile)
    category_probability = model.predict(X_test)
    for cp in category_probability:
        writer.writerow([str(np.argmax(cp))])
    csvfile.close()

# extract_feature(file_name)
# build_model(file_name, weights_file)