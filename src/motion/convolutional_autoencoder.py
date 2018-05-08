import BVH
import tkinter as tk
from tkinter import filedialog
import Animation 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.optimizers import SGD
from Quaternions import Quaternions


FRAME_NUMBER = 250
FILTERS = 64
CONVO_WINDOW = (2, 3)
CONVO_STRIDE = (4, 5)
POOL_REDUCTION = (5, 2)





def load_anim_from_file(filepaths, size):
    x_train = None  
    new_row = False
    for i in filepaths :
        anim = None
        new_line = False
        walk = BVH.load(file_path[0])[0]
        for i in range(0, len(walk.positions)):
            #pos = walk.positions[i]
            #pos = pos.flatten()
            #pos = pos.astype('float32')
            if(anim != None):
                if(len(anim) == FRAME_NUMBER):
                    break
            rotation = walk.rotations[i]
            rotation = rotation.qs
            #rotation = rotation.flatten()
            #frame_data = np.concatenate((pos, rotation))
            frame_data = np.reshape(rotation, (31,4))
            if(anim == None):
                anim = frame_data
            else:
                if(not new_line):
                    anim = np.stack((anim, frame_data))
                    new_line = True
                else:
                    anim = np.concatenate((anim, [frame_data]))

        if(x_train == None):
            x_train = anim
        else:
            if(not new_row):
                x_train = np.stack((x_train, anim))
                new_row = True
            else:
                x_train = np.concatenate((x_train, [anim]))
    return x_train



#Select filepaths
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilenames()

#Determine length of a frame
test, names, frametime = BVH.load(file_path[0])
size = len(test.rotations[0])







#Getting data

data = load_anim_from_file(file_path, size)
print(np.shape(data))
np.random.shuffle(data)
splitted = np.array_split(data, 2)
x_train = splitted[0]
splitted = np.array_split(splitted[1], 2)
x_train = np.concatenate((x_train, splitted[0]))
x_test = splitted[1]

print(len(x_train[0]))
print(np.shape(x_train))


#network construction
input_anim = Input(shape=(FRAME_NUMBER, size, 4))
convo_enc = Conv2D(FILTERS, CONVO_WINDOW, activation = 'relu', padding = 'same')(input_anim)
pool_enc = MaxPooling2D(POOL_REDUCTION, padding = 'same') (convo_enc)

convo_dec = Conv2D(FILTERS, (5,10), activation='relu', padding = 'same')(pool_enc)
pool_dec = UpSampling2D(POOL_REDUCTION)(convo_dec)
decoded = Conv2D(4, CONVO_WINDOW, activation = 'sigmoid', padding = 'same')(pool_dec)


autoencoder = Model(input_anim, decoded)

sgd = SGD(lr=0.0001, decay=1e-5, momentum=0.009, nesterov=True)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


#training
x_yolo = np.empty(((len(x_train), len(x_train[0]), len(x_train[0][0]) + 1 , 4)))
print(np.shape(x_yolo))

for i in range(0,len(x_train)) :
    for j in range(0,len(x_train[i])) :        
        x_yolo[i][j] = np.concatenate((x_train[i][j], [x_train[i][j][len(x_train[i][j])- 1]]))
print(np.shape(x_yolo))

x_yolo2 = np.empty(((len(x_test), len(x_test[0]), len(x_test[0][0]) + 1 , 4)))
for i in range(0,len(x_test)) :
    for j in range(0,len(x_test[i])) :        
        x_yolo2[i][j] = np.concatenate((x_test[i][j], [x_test[i][j][len(x_test[i][j])- 1]]))

autoencoder.fit(x_train, x_yolo,
                epochs=1000,
                batch_size=200,
                shuffle=True, 
                validation_data=(x_test, x_yolo2))


file_path = filedialog.askopenfilenames()
print(file_path)
x_test = load_anim_from_file(file_path, size)
x_test = [x_test]
print(np.shape(x_test))
result_frames = autoencoder.predict(x_test)

print(np.shape(result_frames))



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
result_frames = result_frames[0]
result_frames = np.delete(result_frames, len(result_frames[0]) - 1, 1)
print(np.shape(result_frames))
for i in result_frames :
    if(result_rot == None):
        result_rot =  np.reshape(i, (1, -1, 4))
    else:
        result_rot = np.vstack((result_rot, np.reshape(i, (1, -1, 4))))









test.rotations = Quaternions(result_rot)

BVH.save("../../output.bvh", test, names, frametime, 'zyx', False)
