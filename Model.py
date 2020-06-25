from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten,BatchNormalization, Activation
from tensorflow.keras.layers import Dense,Dropout,MaxPooling2D,MaxPool2D
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gc
import pandas as pd 
import os
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

model = Sequential()

kernel_size = (3,3)
pool_size= (2,2)
first_filters = 64
second_filters = 128
third_filters = 256


# In[8]:


dropout_conv = 0.2
dropout_dense = 0.2


# In[9]:



# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
# from keras.layers import Conv2D, MaxPool2D
model = Sequential()

#now add layers to it

#conv block 1
model.add(Conv2D(first_filters, kernel_size, input_shape = (96, 96, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(first_filters, kernel_size, input_shape = (96, 96, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(first_filters, kernel_size, input_shape = (96, 96, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

#a fully connected (also called dense) layer at the end
model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))

#finally convert to values of 0 to 1 using the sigmoid activation function
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
print(model.summary())
path = ""
model.load_weights("model.h5")

base_test_dir = path + 'test/' #specify test data folder
test_files = glob(os.path.join(base_test_dir,'*.tif')) #find the test file names
submission = pd.DataFrame() #create a dataframe to hold results
file_batch = 10000 #we will predict 10000 images at a time
max_idx = len(test_files) #last index to use
for idx in range(0, max_idx, file_batch): #iterate over test image batches
    print("Indexes: %i - %i"%(idx, idx+file_batch))
    test_df = pd.DataFrame({'path': test_files[idx:idx+file_batch]}) #add the filenames to the dataframe
    test_df['id'] = test_df.path.map(lambda x: x.split('/')[1].split(".")[0]) #add the ids to the dataframe
    test_df['image'] = test_df['path'].map(cv2.imread) #read the batch
    K_test = np.stack(test_df["image"].values) #convert to numpy array
    predictions = model.predict(K_test,verbose = 1) #predict the labels for the test data
    test_df['label'] = predictions #store them in the dataframe
    submission = pd.concat([submission, test_df[["id", "label"]]])
submission.to_csv("submission.csv", index = False, header = True)


# # later...
 
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
