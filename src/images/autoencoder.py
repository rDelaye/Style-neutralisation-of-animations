import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import SGD

aera = 5 #from 0 to 5 X and Y

#define the triangle's vertices
A = np.array([aera*np.random.ranf(), aera*np.random.ranf()])
B = np.array([aera*np.random.ranf(), aera*np.random.ranf()])
C = np.array([aera*np.random.ranf(), aera*np.random.ranf()])

#find the trianlg's edges'equations (y = mx+c)
m1 = (B[1]-A[1])/(B[0]-A[0])
c1 = (B[1]-(m1*B[0]))
if(m1 * C[0] + c1 > C[1]):
    #print('C est en-dessous de AB')
    toC = False
else:
    #print('C est au-dessus de AB')
    toC = True

m2 = (C[1]-B[1])/(C[0]-B[0])
c2 = (C[1]-(m2*C[0]))
if(m2 * A[0] + c2 > A[1]):
    #print('A est en-dessous de BC')
    toA = False
else:
    #print('A est au-dessus de BC')
    toA = True

m3 = (A[1]-C[1])/(A[0]-C[0])
c3 = (A[1]-(m3*A[0]))
if(m3 * B[0] + c3 > B[1]):
    #print('B est en-dessous de CA')
    toB = False
else:
    #print('B est au-dessus de CA')
    toB = True

def one_sample():
    point = np.array([aera*np.random.ranf(), aera*np.random.ranf()])
    #if the point is inside the triangle
    if ((toC == (m1 * point[0] + c1 < point[1])) & (toA == (m2 * point[0] + c2 < point[1])) & (toB == (m3 * point[0] + c3 < point[1])) ):
        value = np.array([0, 1])
        #print('le point est dans le triangle')
    else:
        value = np.array([1, 0])
        #print('le point n\'est pas dans le triangle')
    return point, value


def next_batch(n):
    x = np.zeros(shape=(n,2), dtype=np.float32)
    y = np.zeros(shape=(n,2), dtype=np.int32)
    for i in range(0, n):
        x[i],y[i] = one_sample()
    return x,y


def true_batch(n):
    i = 0
    x = np.zeros(shape=(n,2), dtype=np.float32)
    y = np.zeros(shape=(n,2), dtype=np.int32)
    while i < n :
        x[i],y[i] = one_sample()
        if(y[i][0] == 0) :
            i = i+1
    return x,y

encoding_dim = 1

input_img = Input(shape=(2,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(2, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

x_train, y_train = true_batch(4096)
x_test, y_test = true_batch(4096)
autoencoder.fit(x_train, x_train,
                epochs=200,
                batch_size=128,
                shuffle=True)

encoded_dot = encoder.predict(x_test)
decoded_dot = decoder.predict(encoded_dot)




plt.figure(1)
plt.axis([0, aera, 0, aera])

#draw the triangle's vertices
plt.plot(A[0], A[1], 'ro', color='blue')
plt.plot(B[0], B[1], 'ro', color='blue')
plt.plot(C[0], C[1], 'ro', color='blue')
#draw the triangle's edges
plt.plot([A[0], B[0]], [A[1], B[1]], 'blue')
plt.plot([B[0], C[0]], [B[1], C[1]], 'blue')
plt.plot([C[0], A[0]], [C[1], A[1]], 'blue')

for i in range(256):
    if(np.argmax(x_test[i])==1):
        plt.plot(x_test[i,0], x_test[i,1], 'ro', color='green')
    else:
        plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='red')

#plt.show()




plt.figure(2)
plt.axis([0, aera, 0, aera])

#draw the triangle's vertices
plt.plot(A[0], A[1], 'ro', color='blue')
plt.plot(B[0], B[1], 'ro', color='blue')
plt.plot(C[0], C[1], 'ro', color='blue')
#draw the triangle's edges
plt.plot([A[0], B[0]], [A[1], B[1]], 'blue')
plt.plot([B[0], C[0]], [B[1], C[1]], 'blue')
plt.plot([C[0], A[0]], [C[1], A[1]], 'blue')

for i in range(256):
    if(np.argmax(decoded_dot[i])==1):
        plt.plot(x_test[i,0], x_test[i,1], 'ro', color='green')
    else:
        plt.plot(decoded_dot[i, 0], decoded_dot[i, 1], 'ro', color='red')

#    for i in range(1, 200):
#        dot, dotValue = one_sample()
#        if(dotValue[1] == 0):
#            plt.plot(dot[0], dot[1], 'bs', color='yellow')
#        else:
#            plt.plot(dot[0], dot[1], 'bs', color='green')

plt.show()

x_test, y_test = next_batch(256)
encoded_dot = encoder.predict(x_test)
decoded_dot = decoder.predict(encoded_dot)



plt.figure(3)
plt.axis([0, aera, 0, aera])

#draw the triangle's vertices
plt.plot(A[0], A[1], 'ro', color='blue')
plt.plot(B[0], B[1], 'ro', color='blue')
plt.plot(C[0], C[1], 'ro', color='blue')
#draw the triangle's edges
plt.plot([A[0], B[0]], [A[1], B[1]], 'blue')
plt.plot([B[0], C[0]], [B[1], C[1]], 'blue')
plt.plot([C[0], A[0]], [C[1], A[1]], 'blue')

for i in range(256):
    if(np.argmax(x_test[i])==1):
        plt.plot(x_test[i,0], x_test[i,1], 'ro', color='green')
    else:
        plt.plot(x_test[i, 0], x_test[i, 1], 'ro', color='red')

#plt.show()




plt.figure(4)
plt.axis([0, aera, 0, aera])

#draw the triangle's vertices
plt.plot(A[0], A[1], 'ro', color='blue')
plt.plot(B[0], B[1], 'ro', color='blue')
plt.plot(C[0], C[1], 'ro', color='blue')
#draw the triangle's edges
plt.plot([A[0], B[0]], [A[1], B[1]], 'blue')
plt.plot([B[0], C[0]], [B[1], C[1]], 'blue')
plt.plot([C[0], A[0]], [C[1], A[1]], 'blue')

for i in range(256):
    if(np.argmax(decoded_dot[i])==1):
        plt.plot(x_test[i,0], x_test[i,1], 'ro', color='green')
    else:
        plt.plot(decoded_dot[i, 0], decoded_dot[i, 1], 'ro', color='red')

#    for i in range(1, 200):
#        dot, dotValue = one_sample()
#        if(dotValue[1] == 0):
#            plt.plot(dot[0], dot[1], 'bs', color='yellow')
#        else:
#            plt.plot(dot[0], dot[1], 'bs', color='green')

plt.show()