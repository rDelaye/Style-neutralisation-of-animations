import BVH
import tkinter as tk
from tkinter import filedialog
import Animation 
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.optimizers import SGD
from Quaternions import Quaternions


def load_anim_from_file(filepaths, size):
    x_train = None  
    for i in filepaths :
        walk = BVH.load(file_path[0])[0]
        for i in range(0, len(walk.positions)):
            #pos = walk.positions[i]
            #pos = pos.flatten()
            #pos = pos.astype('float32')
            rotation = walk.rotations[i]
            rotation = rotation.qs
            rotation = rotation.flatten()
            #frame_data = np.concatenate((pos, rotation))
            frame_data = rotation
            if(x_train == None):
                x_train = frame_data
            else:
                x_train = np.vstack((x_train, frame_data))

    '''
        for i in walk.positions :        
            i = i.flatten()
            i = i.astype('float32')
            x_train = np.vstack((x_train, i))
    '''
    return x_train



#Select filepaths
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames()

#Determine length of a frame
test, names, frametime = BVH.load(file_path[0])
size = len(test.positions[0]) * 4


#Getting data

data = load_anim_from_file(file_path, size)
print(np.shape(data))
np.random.shuffle(data)
splitted = np.array_split(data, 2)
x_test = splitted[1]
x_train = splitted[0]


#network construction
input_frame = Input(shape=(size,))
encoded = Dense(size - 20, activation='relu')(input_frame)
#min = Dense(size//2, activation='relu')(encoded)
#dec = Dense(size, activation='relu')(min)
decoded = Dense(size, activation='sigmoid')(encoded)
autoencoder = Model(input_frame, decoded)

sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.009, nesterov=True)
autoencoder.compile(optimizer=sgd, loss='mean_absolute_error')


#training
autoencoder.fit(x_train, x_train,
                epochs=300,
                batch_size=200,
                shuffle=True, 
                validation_data=(x_test, x_test))


file_path = filedialog.askopenfilenames()
print(file_path)
x_test = load_anim_from_file(file_path, size)
print(len(x_test))
result_frames = autoencoder.predict(x_test)

result_pos = None
result_rot = None    
print(len(result_frames))
'''
for i in result_frames :
    if(result_pos == None):
        result_pos = np.reshape(i[0:size//7*3], (1, -1, 3))
    else:
        result_pos = np.vstack((result_pos, np.reshape(i[0:size//7*3], (1, -1, 3))))
    if(result_rot == None):
        result_rot =  np.reshape(i[size//7*3:], (1, -1, 4))
    else:
        result_rot = np.vstack((result_rot, np.reshape(i[size//7*3:], (1, -1, 4))))
'''
for i in result_frames :
    if(result_rot == None):
        result_rot =  np.reshape(i, (1, -1, 4))
    else:
        result_rot = np.vstack((result_rot, np.reshape(i, (1, -1, 4))))




test.rotations = Quaternions(result_rot)

BVH.save("../../output.bvh", test, names, frametime, 'zyx', False)
