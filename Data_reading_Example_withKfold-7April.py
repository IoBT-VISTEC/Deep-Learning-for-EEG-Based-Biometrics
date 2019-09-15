
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.cross_validation import StratifiedKFold
import glob
import numpy as np
import os,sys 
import random
import keras
from keras.models import Sequential, load_model
from keras.layers import Activation,Dense,Dropout,TimeDistributed,Reshape,Flatten,GRU, Conv2D, LSTM 
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from scipy import stats 
from time import time


# In[4]:


'''
------ Variable Deaclaration ------
'''
num_classes = 32
batch_size = 512

# Number of epochs for each training
epochs = 200 
do_rate = 0.3
winlen = 128
step = 128
n_ch=32
# Create list for keeping total accuravy and lost
# This will be used to save as numpy array results
total_acc = []
total_loss = []
training_loss = []
total_time = []
# use for printing
channel_name = ['Frontal','all']
bands_name = ['4-8Hz','8-15Hz','15-32Hz','32-40Hz','all bands']
classes_name = ['H-Valence, H-Arousal','H-Valence, L-Arousal','L-Valence, H-Arousal','L-Valence, L-Arousal','all classes']


#Apiwat -- randomly choose 5 samples for each individual person, called before K-fold   
def datagen(data_x,Y_c,c):
    pp = np.where(Y_c == True)[0]
    temp = np.zeros((len(pp),5,32,6,1280)) 
    np.random.seed(0)
    for i in range(len(pp)):
        t = np.where(data.iloc[:,pp[i]]==c)[0]
        l = np.random.choice(t,5, replace=False)
        temp[i] = data_x[pp[i],l,:,:,:]
    return temp,pp

def generator(features, labels, batch_size):
    batch_features = np.zeros((batch_size, number, 9, 9, win_len))
    batch_labels = np.zeros((batch_size, 1))
    while True:
        for i in range(batch_size):
            index = np.random.randint(features.shape[0])
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels
#build model 9x9 mesh grid, softmax on 'num_classes' global parameter
def GRU_model():
    model = Sequential()
    # first layer
    model.add(TimeDistributed(Conv2D(128,(3,3), padding='same', data_format="channels_last"), 
                              input_shape=(number,9,9,win_len)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(do_rate))

    # hidden layer
    model.add(TimeDistributed(Conv2D(64,(3,3), padding='same')))
    model.add(BatchNormalization())
    model.add(Activation('relu'))                    
    model.add(Dropout(do_rate))

    # hidden layer
    model.add(TimeDistributed(Conv2D(32,(3,3), padding='same')))
    model.add(BatchNormalization())
    model.add(Activation('relu'))     
    model.add(Dropout(do_rate))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation('relu'))
    model.add(Dropout(do_rate))

    # output layer
    # Select between GRU or LSTM
    # Change size of layer follow experiment design
    model.add(LSTM(32, recurrent_dropout=do_rate ,return_sequences=True, implementation=1))
    model.add(Dropout(do_rate))
    model.add(LSTM(16, recurrent_dropout=do_rate ,return_sequences=False, implementation=1))
    model.add(Dropout(do_rate))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print(model.summary()) # Show summary of model
    
    return model
# training, tale X_train,y_train,X_val,_y_val ,returns best model index and training time
def training(X_tr,Y_tr,X_v,Y_v):
    model = GRU_model()
    
    #compute weight, ideally, we don't need this if all classes are equal in number.
    weight = np.zeros(num_classes) # class
    #counting classes
    for i in Y_tr:
        weight[int(i)] +=1 
    print('Training class count')
    print(weight)
    d = np.min([1.0/temp for temp in weight])
    weight = [1.0/temp/d for temp in weight]

    rmsprop = keras.optimizers.rmsprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
    adam = keras.optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=1e-6)
    # OBS rmsprop and adam cant get loss past 2, while sgd got as low as 0.2
  
    model.compile(loss='sparse_categorical_crossentropy', optimizer= rmsprop, metrics=['sparse_categorical_accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1500, min_lr=0.0001)
    csv_logger = CSVLogger('allnode_PIN.log')
    filepath="allnode_weights-v-{epoch:02d}.hdf5"
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=filepath, verbose=1, save_best_only=True)
    start = time()

    model.fit_generator(generator(X_tr,Y_tr,256),steps_per_epoch= 4 ,
                        epochs=epochs,validation_data=([X_v,Y_v]),
                        callbacks=[checkpointer,csv_logger,reduce_lr],
                        max_queue_size=3,
                        use_multiprocessing=True,class_weight=weight)   
    end = time()
    print('time:',end - start)

    '''
    open log file to find best weight
    '''

    # open log file
    # log file contains loss of each epoch in each running step
    log_file = "./allnode_PIN.log"
    loss = []
    with open(log_file) as f:
        f = f.readlines()
    f[0:] = f[1:] #delete header
    for line in f:
        loss.append([int(line.split(',')[0]), float(line.split(',')[3])]) #save all lost to list

    # find minimum loss
    # Keep minimum lost and save its index and use it for testing  
    min_loss = 100
    for data in range(len(loss)):
        if loss[data][1] < min_loss:
            min_loss = loss[data][1]
            best_model_index = loss[data][0]
    return best_model_index,end-start,loss[:][1]


'''
------ Testing part ------
'''
num_classes = 32
do_rate = 0.3

win_len = 128 # 1s
number=10
step=128
#take X_test,y_test,path_to hdf5 file(string), yields loss and acc,model
def testing(X_te, Y_te, w_path):
    model = load_model(w_path)
    rmsprop = keras.optimizers.rmsprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='sparse_categorical_crossentropy', optimizer= rmsprop, metrics=['sparse_categorical_accuracy'])
    loss,acc = model.evaluate(X_te, Y_te, batch_size=256, verbose=1, sample_weight=None)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    # Save accuracy and loss to list
    return loss,acc,model


#files are separeted by frequency bands
X = np.zeros((1,32,40,32,6,1280)) 
#old format, X = np.zeros((1,32,30,32,1280)) # (frequency bands,person,trials,channels,time(10s))
for band in range(1):
    X[band] = np.load('X_'+str(4)+'.npy') #last index is all bands, others are in incresing order
        

#this code block is to obtain list of subject to be used by each valence-arousal class(>5 people)
import pandas as pd
Y = np.load('Y_4classes.npy')
Y_ = Y[:,:,4].reshape((32,-1))
data = pd.DataFrame(np.swapaxes(Y[:,:,4].reshape((32,-1)),0,1))
df=data.apply(pd.value_counts)
Y_cut =(df.append(pd.DataFrame(np.repeat(40,32).reshape(1,32)))>5).as_matrix()
#del data,df,Y_ ,Y


# In[8]:


for band in range(1):   
    for classes in range(5):    
        #old -- temp_x = X[band,Y_cut[classes]] #a band,people,30,32,1280
        temp_x,y_list = datagen(X,Y_cut[classes],classes)
        temp_x = np.swapaxes(temp_x,-2,-3)
        temp_x = temp_x.reshape((temp_x.shape[0],5,6,32,1280))
        #temp_x = temp_x.reshape(sum(Y_cut[classes])*30,32,1280) #sum(Y_cut[classes]) is total number of people
         # Sliding array

        
        x_slice = np.zeros((sum(Y_cut[classes]), 5,6, 10, 32, 128))
        for slice_num in range(10): #num of slice
            start_pos = slice_num * step
            stop_pos = start_pos + win_len
            x_slice[:, :,:, slice_num, :, :]=temp_x[:,:,:,:,start_pos:stop_pos]

        #prevent memory shortage
        del temp_x
        print("X_slice input shape: ",x_slice.shape)

        #Normalization
        # max = np.amax(x_slice, axis=3) 
        # min = np.amin(x_slice, axis=3) 
       
        print("X_slice new shape: ",x_slice.shape)
        print('Creating the 2D mesh...')
        # map the channels into 2D mesh
        x_2d = np.zeros((sum(Y_cut[classes]), 5,6, 10, 9, 9, win_len))
        channel_2d = [[0,3],[1,3],[2,2],[2,0],[3,1],[3,3],[4,2],[4,0],[5,1],[5,3],[6,2],[6,0],[7,2],[8,3],[8,4],[6,4],[0,5],[1,5],[2,4],
                      [2,6],[2,8],[3,7],[3,5],[4,4],[4,6],[4,8],[5,7],[5,5],[6,6],[6,8],[7,6],[8,5]]
        x_2d[:,:,:,:,0,3,:] = x_slice[:,:,:,:,0,:]
        x_2d[:,:,:,:,1,3,:] = x_slice[:,:,:,:,1,:]
        x_2d[:,:,:,:,2,2,:] = x_slice[:,:,:,:,2,:]
        x_2d[:,:,:,:,2,0,:] = x_slice[:,:,:,:,3,:]
        x_2d[:,:,:,:,3,1,:] = x_slice[:,:,:,:,4,:]
        x_2d[:,:,:,:,3,3,:] = x_slice[:,:,:,:,5,:]
        x_2d[:,:,:,:,4,2,:] = x_slice[:,:,:,:,6,:]
        x_2d[:,:,:,:,4,0,:] = x_slice[:,:,:,:,7,:]
        x_2d[:,:,:,:,5,1,:] = x_slice[:,:,:,:,8,:]
        x_2d[:,:,:,:,5,3,:] = x_slice[:,:,:,:,9,:]
        x_2d[:,:,:,:,6,2,:] = x_slice[:,:,:,:,10,:]
        x_2d[:,:,:,:,6,0,:] = x_slice[:,:,:,:,11,:]
        x_2d[:,:,:,:,7,2,:] = x_slice[:,:,:,:,12,:]
        x_2d[:,:,:,:,8,3,:] = x_slice[:,:,:,:,13,:]
        x_2d[:,:,:,:,8,4,:] = x_slice[:,:,:,:,14,:]
        x_2d[:,:,:,:,6,4,:] = x_slice[:,:,:,:,15,:]
        x_2d[:,:,:,:,0,5,:] = x_slice[:,:,:,:,16,:]
        x_2d[:,:,:,:,1,5,:] = x_slice[:,:,:,:,17,:]
        x_2d[:,:,:,:,2,4,:] = x_slice[:,:,:,:,18,:]
        x_2d[:,:,:,:,2,6,:] = x_slice[:,:,:,:,19,:]
        x_2d[:,:,:,:,2,8,:] = x_slice[:,:,:,:,20,:]
        x_2d[:,:,:,:,3,7,:] = x_slice[:,:,:,:,21,:]
        x_2d[:,:,:,:,3,5,:] = x_slice[:,:,:,:,22,:]
        x_2d[:,:,:,:,4,4,:] = x_slice[:,:,:,:,23,:]
        x_2d[:,:,:,:,4,6,:] = x_slice[:,:,:,:,24,:]
        x_2d[:,:,:,:,4,8,:] = x_slice[:,:,:,:,25,:]
        x_2d[:,:,:,:,5,7,:] = x_slice[:,:,:,:,26,:]
        x_2d[:,:,:,:,5,5,:] = x_slice[:,:,:,:,27,:]
        x_2d[:,:,:,:,6,6,:] = x_slice[:,:,:,:,28,:]
        x_2d[:,:,:,:,6,8,:] = x_slice[:,:,:,:,29,:]
        x_2d[:,:,:,:,7,6,:] = x_slice[:,:,:,:,30,:]
        x_2d[:,:,:,:,8,5,:] = x_slice[:,:,:,:,31,:]
    
        del x_slice

        x_2d = x_2d.reshape(sum(Y_cut[classes])*5,6, 10, 9, 9, win_len) #(1280,10,9,9,128)
        temp_y = np.repeat(y_list,5)
        #y = np.repeat(np.arange(sum(Y_cut[classes])),30)
        np.random.seed(0) #reset random seed for each band, each class
        skf = StratifiedKFold(y,n_folds=5, shuffle=True, random_state=None)
        mylist = np.array(list(skf))
        kArray = mylist[:,1]
        for k in range(5):
            train_index = np.concatenate([kArray[i] if i!=k else [] for i in range(5)])
            test_index = kArray[k]
            train_index = train_index.astype(int)
            test_index = test_index.astype(int)
            X_train, X_val, X_test = x_2d[train_index[:-int(len(train_index)/4)]], x_2d[train_index[-int(len(train_index)/4):]],x_2d[test_index]
            y_train, y_val, y_test = y[train_index[:-int(len(train_index)/4)]], y[train_index[-int(len(train_index)/4):]], y[test_index]
            
            X_train = np.reshape(X_train,(-1,10,9,9,128))
            X_val = np.reshape(X_val,(-1,10,9,9,128))
            X_test = np.reshape(X_test,(-1,10,9,9,128))
            y_train = np.repeat(y_train,6)
            y_val = np.repeat(y_val,6)
            y_test = np.repeat(y_test,6)

            num_classes = sum(Y_cut[classes])
            
            #normalize
            mu=[stats.tmean(X_train[:,:,d[0],d[1]])  for d in channel_2d] # (32,30,10,32,128)
            std=[stats.tstd(X_train[:,:,d[0],d[1]])for d in channel_2d]
            for i in range(len(channel_2d)):
                X_train[:,:,channel_2d[i][0],channel_2d[i][1]]= ((X_train[:,:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i])
                X_val[:,:,channel_2d[i][0],channel_2d[i][1]]= ((X_val[:,:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i])
                X_test[:,:,channel_2d[i][0],channel_2d[i][1]]= ((X_test[:,:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i])
                
            print("---------------------------------------------")
            print('Start training  --- classes :%s, Bands: %s \n\n' % (classes_name[classes],bands_name[band]))
            print("---------------------------------------------")
            #training
            best_model_index,t,temp_loss = training(X_train,y_train,X_val,y_val)
            total_time.append([band,classes,k,t])
            #testing
            loss, acc, model_ = testing(X_test,y_test,'allnode_weights-v-'+str(int(best_model_index)+1)+'.hdf5',)
            total_loss.append([band,classes,k,loss])
            total_acc.append([band,classes,k,acc])
            training_loss.append([band,classes,k,temp_loss])
            #model_.save('band_%d_classes_%d_fold_%d.h5' %(band,classes,k))
            # Show testing results
            print("---------------------------------------------")
            print('Allnode END of part --- classes :%s, Bands: %s \n\n' % (classes_name[classes],bands_name[band]))
            print("---------------------------------------------")

            # Save accuracy and loss list to numpy array
            np.save("Allnode_Total_acc.npy",total_acc)
            np.save("Allnode_Total_loss.npy",total_loss)
            np.save("Allnode_Total_time.npy",total_time)
            np.save("Allnode_training_loss.npy",training_loss)

