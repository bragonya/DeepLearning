from keras import backend as K
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.layers import LeakyReLU, ELU
import numpy as np
import matplotlib.pyplot as plt

dictionary = {0:'airplane',1:'autommovil',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
def getLabelName(index):
    return dictionary.get(yyt[index][0])
    
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

def normalization(dataset):
    mean = np.mean(x_train,axis=(0,1,2,3))
    desviation = np.std(x_train,axis=(0,1,2,3))
    return (dataset-mean)/(desviation+ K.epsilon())

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
yyt = y_train.copy()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
 
x_train = normalization(x_train)
x_test = normalization(x_test)
 
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
model = Sequential()
model.add(Conv2D(32, (3,3),kernel_regularizer=regularizers.l2(1e-4),input_shape=x_train.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3),kernel_regularizer=regularizers.l2(1e-4)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Conv2D(64, (3,3),kernel_regularizer=regularizers.l2(1e-4)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3),kernel_regularizer=regularizers.l2(1e-4)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(10, activation='softmax'))
 
model.summary()
 
#data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
    )
datagen.fit(x_train)
batch_size = 64
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size,epochs=50,verbose=1,validation_data=(x_test,y_test))

model.save_weights('modelwithoutDropout2layersRegulizer.h5') 

print(getLabelName(11))
plt.imshow(x_train[11])
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
            
activations = activation_model.predict(x_train[11].reshape(1,32,32,3))
display_activation(activations, 8, 4, 2)

scores = model.evaluate(x_test, y_test)
print(scores)
