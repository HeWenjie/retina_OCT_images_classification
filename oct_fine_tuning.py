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

def fine_tuning(train_path, layer_num, image_size=(299, 299)):

    img_gen = ImageDataGenerator(rescale=1./255)
    train_generator = img_gen.flow_from_directory(
            train_path,
            target_size=image_size,
            class_mode='categorical',
            classes=['AMD','DME','NOR'],
            shuffle=False)

    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:layer_num]:
        layer.trainable = False
    for layer in model.layers[layer_num:]:
        layer.trainable = True

    model.compile(optimizer='Adadelta', loss='categorical_crossentropy',metrics=['categorical_accuracy'])

    for i in range(50):
        print ("round: " + str(i + 1))
        model.fit_generator(train_generator)
        train_acc = model.evaluate_generator(train_generator)
        print('train_acc = '  + str(train_acc[1]))
    
    model.save(model_file)

def fine_tuning_predict(test_path, model_file, result_path, image_size=(299, 299)):

    img_gen = ImageDataGenerator(rescale=1./255)
    test_generator = img_gen.flow_from_directory(
            test_path,
            target_size=image_size,
            class_mode='categorical',
            classes=['AMD','DME','NOR'],
            shuffle=False)

    model = load_model(model_file)

    test_acc = model.evaluate_generator(test_generator)
    print ('Test Accuarcy: ' + str(test_acc[1]))

    csvfile = open(result_path, "w")
    writer = csv.writer(csvfile)
    category_probability = model.predict_generator(test_generator)
    for cp in category_probability:
        writer.writerow([str(np.argmax(cp))])
    csvfile.close()
