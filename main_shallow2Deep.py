
import scipy.io as sio
import numpy as np
from SAR_utils import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from model_shallow2Deep import cmplx_shallow2Deep, real_shallow2Deep, cmplx_shallow2Deep_SE
from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

# Get the data
dataset = 'FL'
windowSize = 11 # int(input("Enter the window size\n"))
train_per = 1 # int(input("Enter the percentage of training data:\n"))
coh_data, cov_data, labels = loadData(dataset)




X_coh, y = createImageCubes(coh_data, labels, windowSize)


X_train, X_test, y_train, y_test = splitTrainTestSet(X_coh, y, 1-train_per/100, randomState=345)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

Model = cmplx_shallow2Deep_SE(X_train, num_classes(dataset))
Model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )


    
history = Model.fit(X_train, y_train,
                            batch_size = 16, 
                            verbose = 1, 
                            epochs = 100, 
                            shuffle = True,
                            callbacks = [early_stopper] )
    
#Model.save_weights('./Models_Weights/'+ dataset +'/cmplx_SE_winSize_' + str(windowSize) + '_iter_' + str(i)+'.h5')
    
Y_pred_test = Model.predict([X_test])
y_pred_test = np.argmax(Y_pred_test, axis=1)
       
    
    
    
kappa = cohen_kappa_score(np.argmax(y_test, axis=1),  y_pred_test)
oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)

print("Overall Accuracy =",  oa)
print("Average Accuracy =", aa)
print("Kappa =", kappa) 


###############################################################################
# Create the predicted class map
X_coh, y = createImageCubes(coh_data, labels, windowSize, removeZeroLabels = False)
Y_pred_test = Model.predict([X_coh])
y_pred_test = (np.argmax(Y_pred_test, axis=1))
Y_pred = np.reshape(y_pred_test, labels.shape) + 1
gt_binary = np.copy(labels)
gt_binary[labels>0]=1
outputs = Y_pred*gt_binary

sio.savemat('cmplx_shallow2deep_SE_' + dataset +'.mat', {'outputs': outputs})


plt.figure()
plt.imshow(outputs,  cmap=plt.cm.get_cmap("jet", np.max(labels+1))), plt.colorbar()
plt.title("Predicted Class Map")

plt.figure()
plt.imshow(y,  cmap=plt.cm.get_cmap("jet", np.max(labels+1))), plt.colorbar()
plt.title("Reference Class Map")
