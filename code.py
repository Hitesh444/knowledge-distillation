import tensorflow as tf
import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.datasets import cifar10
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Lambda, concatenate, Activation, Flatten
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import argparse
from keras import regularizers
from keras import optimizers
import h5py
temperature=5

def normalize(X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test
    
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train,x_test=normalize(x_train,x_test)

##just written to check the working


def model():
    #model=keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(32,32,3), pooling=None, classes=1000)
            # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',input_shape=[32,32,3],kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    
    model.load_weights('cifar10vgg.h5', by_name=False)
    #weights=model.layers[-1].get_weights()
    layer_last=model.layers.pop()
    #print(np.array(layer_last.get_weights()).shape)
    #model.add()
    ## usual probabilities
    #logits = model.layers[-1].output
    #probabilities = Activation('softmax')(logits)

    ## softed probabilities
    #output=(model.layers[-1].output)
    #layer_new=keras.layers.Dense(10,activation='sigmoid')(output)
    #layer_new.set_weights(weights)
    #model=Model(model.input,layer_new)
    #model.layers[-1].set_weights(weights)
    
    logits = model.layers[-1].output
    probabilities = Activation('softmax')(logits)
    logits_T = Lambda(lambda x: x/temperature)(logits)
    probabilities_T = Activation('softmax')(logits_T)
    output = concatenate([probabilities, probabilities_T],axis = 0)
    model = Model(model.input, output)
    #model.summary()
    '''model.compile(
        #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
        optimizer='adadelta',
        loss='mse',
        metrics=['acc'] #[acc, categorical_crossentropy, soft_logloss]
    )
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=5,
              verbose=1,
              validation_data=(x_test, y_test),
              )
'''
    


model()  
#keras.__version__


def student_model():
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',input_shape=[32,32,3],kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #changed neurons from 512 to 256
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #removed this convolution layer
    '''
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    '''

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    #removed this convolution layer
    '''
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    '''
    
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    #changed neurons to 256 from 512
    model.add(Dense(256,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    layer_last=model.layers.pop()
    learning_rate = 0.1
    lr_decay = 1e-6
    lr_drop = 20
    
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

    sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    
    return model
 
    def myGenerator(dataset, batch_size):
        while 1:
                n = len(dataset)
                steps = int(n/batch_size)
                for i in range(steps):
                    yield i, dataset[i*batch_size:(i+1)*batch_size]

def save_logits(outpaths=('train_logits.npy', 'test_logits.npy'), batch_size=100):
    model = teacher_model()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train,x_test=normalize(x_train,x_test)
    train_logits = []
    train_generator = myGenerator(x_train, batch_size)
    n = len(x_train)
    total_steps = int(n / batch_size)
    print("Processing train data logits in %d steps by batches of size %d" % (total_steps, batch_size))
    for num_batch, x_batch in train_generator:
        print("processing batch %d..." % num_batch)
        batch_logits = model.predict_on_batch(x_batch)

        for i in range(len(batch_logits)):
            #train_logits[num_batch*batch_size+i] = batch_logits[i]
            train_logits.append(batch_logits[i])

        if num_batch >= total_steps-1:
            break

    np.save(outpaths[0], train_logits)
    print('Train logits saved to %s' % outpaths[0])

    test_logits = []
    test_generator = myGenerator(x_test, batch_size)
    n = len(x_test)
    total_steps = int(n / batch_size)
    print("Processing test data logits in %d steps by batches of size %d" % (total_steps, batch_size))
    for num_batch, x_batch in test_generator:
        print("processing batch %d..." % num_batch)

        batch_logits = model.predict_on_batch(x_batch)

        for i in range(len(batch_logits)):
            test_logits.append(batch_logits[i])

        if num_batch >= total_steps-1:
            break


    np.save(outpaths[1], test_logits)
    print('Test logits saved to %s' % outpaths[1])
    
def train(model, training_data, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test)= training_data
    x_train,x_test=normalize(x_train,x_test)
    nb_classes=10
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    '''
    STAMP = model_label
    print('Training model {}'.format(STAMP))
    logs_path = './logs/{}'.format(STAMP)

    bst_model_path = './checkpoints/' + STAMP + '.h5'
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
    tensor_board = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1)
    '''
   
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              shuffle=True,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    """
    model_yaml = model.to_yaml()
    with open("bin/"+STAMP+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/'+STAMP+'model.h5')
    """

def train_student(model,training_data,
                  logits_paths=('train_logits.npy', 'test_logits.npy'),
                  batch_size=256, epochs=10, temp=5.0, lambda_weight=0.1):
    temperature = temp
    (x_train, y_train), (x_test, y_test) = training_data
    nb_classes=10
    x_train,x_test=normalize(x_train,x_test)
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    # load or calculate logits of trained teacher model
    train_logits_path = logits_paths[0]
    test_logits_path = logits_paths[1]
    if not (os.path.exists(train_logits_path) and os.path.exists(test_logits_path)):
        save_logits(logits_paths)
    train_logits = np.load(train_logits_path)
    test_logits = np.load(test_logits_path)

    # concatenate true labels with teacher's logits
    y_train = np.concatenate((y_train, train_logits), axis=1)
    y_test = np.concatenate((y_test, test_logits), axis=1)

    # remove softmax
    model.layers.pop()
    # usual probabilities
    logits = model.layers[-1].output
    probabilities = Activation('softmax')(logits)

    # softed probabilities
    logits_T = Lambda(lambda x: x / temperature)(logits)
    probabilities_T = Activation('softmax')(logits_T)

    output = concatenate([probabilities, probabilities_T], axis = 0)
    model = Model(model.input, output)
    # now model outputs 26+26 dimensional vectors

    #internal functions
    def knowledge_distillation_loss(y_true, y_pred, lambda_const):
        # split in
        #    onehot hard true targets
        #    logits from teacher model
        y_true, logits = y_true[:, :nb_classes], y_true[:, nb_classes:]

        # convert logits to soft targets
        y_soft = K.softmax(logits / temperature)

        # split in
        #    usual output probabilities
        #    probabilities made softer with temperature
        y_pred, y_pred_soft = y_pred[:, :nb_classes], y_pred[:, nb_classes:]

        return lambda_const * logloss(y_true, y_pred) + logloss(y_soft, y_pred_soft)

    # For testing use usual output probabilities (without temperature)
    def acc(y_true, y_pred):
        y_true = y_true[:, :nb_classes]
        y_pred = y_pred[:, :nb_classes]
        return categorical_accuracy(y_true, y_pred)

    def categorical_crossentropy(y_true, y_pred):
        y_true = y_true[:, :nb_classes]
        y_pred = y_pred[:, :nb_classes]
        return logloss(y_true, y_pred)

    # logloss with only soft probabilities and targets
    def soft_logloss(y_true, y_pred):
        logits = y_true[:, nb_classes:]
        y_soft = K.softmax(logits / temperature)
        y_pred_soft = y_pred[:, nb_classes:]
        return logloss(y_soft, y_pred_soft)

    lambda_const = lambda_weight

    model.compile(
        #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
        optimizer='adadelta',
        loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, lambda_const),
        metrics=[acc] #[acc, categorical_crossentropy, soft_logloss]
    )

    '''
    STAMP = model_label
    print('Training model {}'.format(STAMP))
    logs_path = './logs/{}'.format(STAMP)

    bst_model_path = './checkpoints/' + STAMP + '.h5'
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True,
                                                       verbose=1)
    tensor_board = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True,
                                               write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1)
    '''
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # Offload model to file
    model_yaml = model.to_yaml()
    with open("bin/"+STAMP+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'bin/'+STAMP+'model.h5')
    
    
if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-f', '--file', type=str, help='Path .mat file data',
                        required=True) #default='data/matlab/emnist-digits.mat'
    parser.add_argument('-m', '--model', type=str, help='model to be trained (cnn, mlp or student).'
                                                        ' If student is selected than path to pretrained teacher must be specified in --teacher parameter',
                        required=True)
    parser.add_argument('-t', '--teacher', type=str, help='path to .h5 file with weight of pretrained teacher model'
                                                          ' (e.g. bin/cnn_64_128_1024_30model.h5)',
                        default='checkpoints/10cnn_32_64_128_12.h5')

    parser.add_argument('--width', type=int, default=28, help='Width of the images')
    parser.add_argument('--height', type=int, default=28, help='Height of the images')
    parser.add_argument('--max', type=int, default=None, help='Max amount of data to use')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs to train on')
    #parser.add_argument('--verbose', action='store_true', default=False, help='Enables verbose printing')
    args = parser.parse_args()

    """
    
    '''
    bin_dir = os.path.dirname(os.path.realpath(__file__)) + '/bin'
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
    '''

    training_data = cifar10.load_data() #args.verbose)

    
    #elif args.model=='student':
        #label = '10mlp_%d_%d' % (32, args.epochs)
    model = student_model() #build_mlp(#training_data) #args.verbose)
    train(model, training_data, 10)
    #liargs.model=='distill':
    model = model#build_mlp(training_data)  # args.verbose)
    temp = 2.0
    lamb = 0.5
        #label = '10student_mlp_%d_%d_lambda%s_temp%s' % (32, args.epochs, str(lamb), str(temp))

    train_student(model,training_data, epochs=10, temp=temp, lambda_weight=lamb)
    #else:
     #   print('Unknown --model parameter (must be one of these: cnn/mlp/student)!')




