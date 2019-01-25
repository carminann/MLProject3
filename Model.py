import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense
from keras.layers import Flatten
A=np.loadtxt('train4.txt.gz')#carica i caratteri presenti in exercise1
#Riformattare i dati in modo da andare in un tensore 128x128x1
X=A.reshape((4338,128,128,1))#Riorganizzati i dati
A=np.loadtxt('train1.txt.gz')#Dati per la validazione
#Bisogna trasformare le classi in modo da essere compatibili con l'ultimo strato della rete neurale cioè abbiano valori tra 0 e 1
y=A[:,-1]
y=y-1;
model=Sequential()#creazione del modello
model.add(Conv2D(16,5,activation='relu',input_shape=(128,128,1),strides=2))#creazione del primo livello, shape=2 meno neuroni
#Inserimento secondo strato convoluzionale
model.add(Conv2D(32,3,activation='relu'))
model.add(MaxPooling2D(5))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))#zero maschio 1 femmina
model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
h=model.fit(X,y,epochs=10)#Se non si usa epooch di default usa 10
#Se il risultato del training set non è soddisfacente => il modello deve essere più complicato quindi aumentare il numero di neuroni
#o di strati
#Usare una rete più complessa per vedere se migliorano le prestazioni
A=np.loadtxt('test4.txt.gz')
Y1=A.reshape((4338,128,128,1))
#Provare a fare il reshape anche di questo
y1=model.predict(Y1)#risposta sul test set
y1=y1.reshape((4338,))
#Sapere gli errori del sistema
print("Errore: ",(y1!=y).sum()/4338)
