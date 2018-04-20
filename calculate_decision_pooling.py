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

list_of_train = [1, 2, 3, 4, 5, 6, 7, 8]
list_of_test = [9, 10, 11, 12, 13, 14, 15]

def extract_feature(train_path, file_name, image_size=(299, 299)):

    img_gen = ImageDataGenerator(rescale=1./255)
    train_generator = img_gen.flow_from_directory(
            test_folder_name + '/train',
            target_size=image_size,
            class_mode='categorical',
            classes=['AMD','DME','NOR'],
            shuffle=False)
    train_generator = img_gen.flow_from_directory(
            test_folder_name + '/train',
            target_size=image_size,
            class_mode='categorical',
            classes=['AMD','DME','NOR'],
            shuffle=False)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed8').output)

    train = model.predict_generator(train_generator)

    f = h5py.File(file_name, 'w')
    f.create_dataset("train", data=train)
    f.create_dataset("train_label", data=train_generator.classes)
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

def predict_label(model_file, list_of_test, test_folder_name):

    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed8').output)

    gen = ImageDataGenerator(rescale=1./255)

    csvfile = open('decision.csv', 'w')
    writer = csv.writer(csvfile)

    for index in list_of_test:
        path = test_folder_name + '/test/test' + str(index)
        test_generator = gen.flow_from_directory(
            path,
            target_size = image_size,
            class_mode='categorical',
            classes=['AMD' + str(index), 'DME' + str(index), 'NORMAL' + str(index),],
            shuffle=False)

        X_test = np.array(model.predict_generator(test_generator))
        y_test = np.array(test_generator.classes)
        y_test = np_utils.to_categorical(y_test)

        predict_result = my_model.predict(X_test)

        result = []

        base_num = 0
        for name in ['AMD', 'DME', 'NORMAL']:
            num = count_img(path + '/' + name + str(index))
            class_predict_result = predict_result[base_num:base_num+num]
            base_num = base_num + num

            label = []
            for c in class_predict_result:
                label.append(np.argmax(c))
            result.append(np.argmax(np.bincount(label)))

        writer.writerow([str(result[0]), str(result[1]), str(result[2])])
    
    csvfile.close()
            

def count_img(path):
    return len(os.listdir(path))



OCT_FILE = 'OCT1'

# write_gap(InceptionV3, OCT_FILE)
my_model = build_model('gap_InceptionV3.h5', 128, 'inception_weights.h5')
predict_label(InceptionV3, my_model, list_of_test, OCT_FILE)