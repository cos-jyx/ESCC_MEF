# # # -*- coding: utf-8 -*
from __future__ import division,print_function,absolute_import
import os
# CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["PYTHONHASHSEED"] = '0'
# GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["SM_FRAMEWORK"] = "tf.keras"


import tensorflow as tf
# tf.random.set_seed(42)
from resnet3d import Resnet3DBuilder
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv3D,MaxPool3D,BatchNormalization,Input

from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import ReduceLROnPlateau,TensorBoard,EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
# np.random.seed(42)
from tqdm import tqdm
import SimpleITK as sitk
import random
# random.seed(42)
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K
def seed_tensorflow(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_tensorflow()
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def model_evalu(model, data, label,optimal_th):
    predict = np.squeeze(model.predict(data))
    pre_prob_1 = np.squeeze(model.predict(data))
    print(predict)
    predict[np.where(predict >= optimal_th)] = 1
    predict[np.where(predict < optimal_th)] = 0
    # print('model accuracy is'+str(accuracy_score(label, predict)))
    # print('model precision is'+str(precision_score(label, predict)))
    # print('model recall is'+str(recall_score(label, predict)))
    # print('model f1_score is'+str(f1_score(label, predict)))
    # print('model auc is'+str(roc_auc_score(label, pre_prob_1)))
    # print(classification_report(label_test,train_prediction,target_names=['no perforation','perforation']))
    confusion = confusion_matrix(label, predict)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # print(confusion)
    # print('Accuracy:', (TP+TN)/float(TP+TN+FP+FN))
    # print('Sensitivity:', TP / float(TP+FN))
    # print('Specificity:', TN / float(TN+FP))
    return roc_auc_score(label, pre_prob_1),(TP+TN)/float(TP+TN+FP+FN),TP / float(TP+FN)

def plot_roc(model, data_tra, data_test, data_vali, label_tra, label_test, label_vali,file):
    Font = {'size': 15}
    fpr1, tpr1, thres1 = metrics.roc_curve(label_tra, np.squeeze(model.predict(data_tra)),pos_label=1)
    fpr2, tpr2, thres2 = metrics.roc_curve(label_test, np.squeeze(model.predict(data_test)),pos_label=1)
    fpr3, tpr3, thres3 = metrics.roc_curve(label_vali, np.squeeze(model.predict(data_vali)), pos_label=1)
    roc_auc1 = roc_auc_score(label_tra, np.squeeze(model.predict(data_tra)))
    roc_auc2 = roc_auc_score(label_test, np.squeeze(model.predict(data_test)))
    roc_auc3 = roc_auc_score(label_vali, np.squeeze(model.predict(data_vali)))
    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, label='Train = %0.3f' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, label='Test  = %0.3f' % roc_auc2, color='blue')
    plt.plot(fpr3, tpr3, label='Vali  = %0.3f' % roc_auc3, color='green')
    plt.legend(loc='lower right', prop=Font)
    plt.title(file)
    plt.plot([-0.1, 1.1], [-0.1, 1.1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)
    plt.show()
#
# # Parameters Definition
num_epoch = 50#150
learning_rate = 0.0001  # 0.0001
batch_size =16#64
loss_function = 'binary_crossentropy'  # "binary_crossentropy"
optimizer = 'adam'#"adam"
class_weight =  {0:0.551,1:5.357}#{0:0.01,1:0.6}

# Optimizer Definiton
sgd=optimizers.SGD(lr=learning_rate)
adam=tf.optimizers.Adam(lr=learning_rate)

# Reading Label Data
df=pd.read_csv('/data/data2/jyx/chuankong/deep_0701/label.csv',usecols=[0,1])
train_label = df.values[:, 0]
vali_label = df.values[0:52, 1]

# Reading Training data
label_train0=[]
image0 = np.zeros(shape=(0,10,32,32))

for i in tqdm(range(1,675),desc='Reading data:'):
    for j in [0]:
        try:
            nii_path = '/data/data2/jyx/chuankong/deep_0701/train_roi_0_aug/'+'%03d'%i+'_aug_'+str(j)+'.nii'
            # print(nii_path)
            # nii_path = 'D:/task_grade1/cps/train_crop_0/X-'+'%03d'%i+'_crop.nii.gz'
            nii_img = sitk.ReadImage(nii_path)
            array_img = sitk.GetArrayFromImage(nii_img)
            array_img = (array_img - np.min(array_img)) / (np.max(array_img) - np.min(array_img))###最大最小归一
            # array_img = StandardScaler().fit_transform(array_img)
            image0 = np.append(image0, np.expand_dims(array_img, axis=0), axis=0)
            index = int(nii_path.split('/')[-1].split('_')[0])-1
            # print(index,df.values[:, 1][index])
            # label_train.append(df.values[index, 0])
            # print(int(train_label[index]))
            label_train0.append(int(train_label[index]))

        except:
            print(i,j)
data_train0=np.array(image0,dtype=np.float32)
label_train0 = np.array(label_train0)
# label_train0 = to_categorical( , num_classes=3)

# Split Training data and Test data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=475)
for train_index, test_index in split.split(data_train0, label_train0):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = data_train0[train_index], data_train0[test_index]
    y_train, y_test = label_train0[train_index], label_train0[test_index]
# X_train,X_vali,Y_train,Y_vali = train_test_split(data_train,label_train,test_size=0.3,random_state=475,stratify=label_train)

# train  Augmentation
image = np.zeros(shape=(0,10,32,32))
label_train=[]

for i in tqdm(train_index+1, desc='Augment data:'):
    #Enhance only data labelled 1
    if train_label[i-1] == 0:
        random.seed(i)
        a = random.randint(1, 9)
        list = [0,a]
    else:
        random.seed(i)
        a = random.randint(1,9)
        random.seed(i+1)
        b = random.randint(1,9)
        random.seed(i+2)
        c = random.randint(1,9)
        random.seed(i + 3)
        d = random.randint(1, 9)
        random.seed(i + 4)
        e = random.randint(1, 9)
        random.seed(i + 4)
        f = random.randint(1, 9)
        random.seed(i + 4)
        g = random.randint(1, 9)
        list = [0, a, b, c, d,e,f,g ]#, c]  # 0 is the original image, and then n randomly selected enhanced images
    for j in list:
        try:
            nii_path = '/data/data2/jyx/chuankong/deep_0701/train_roi_0_aug/'+'%03d'%i+'_aug_'+str(j)+'.nii'
            nii_img = sitk.ReadImage(nii_path)
            array_img = sitk.GetArrayFromImage(nii_img)
            array_img = (array_img - np.min(array_img)) / (np.max(array_img) - np.min(array_img))###最大最小归一
            image = np.append(image, np.expand_dims(array_img, axis=0), axis=0)
            index = int(nii_path.split('/')[-1].split('_')[0])-1
            label_train.append(int(train_label[index]))

        except:
            print(i, j)
data_train = np.array(image,dtype=np.float32)
label_train = np.array(label_train)

print(data_train.shape)
print(label_train)
# Reading Label Data
label_vali = []
image1 = np.zeros(shape=(0,10,32,32))

for i in tqdm(range(1,54),desc='Reading test data:'):
    try:
        if i == 11:
            continue
        nii_path = '/data/data2/jyx/chuankong/deep_0701/test_roi_0/'+'%03d'%i+'.nii'
        nii_img1 = sitk.ReadImage(nii_path)
        array_img1 = sitk.GetArrayFromImage(nii_img1)
        array_img1 = (array_img1 - np.min(array_img1)) / (np.max(array_img1) - np.min(array_img1))###最大最小归一
        image1 = np.append(image1, np.expand_dims(array_img1, axis=0), axis=0)
    except:
        print(i)
data_vali = np.array(image1,dtype=np.float32)
label_vali = vali_label
label_vali= np.array(vali_label).astype(int)
# label_vali = to_categorical(label_vali, num_classes=3)
#
np.random.seed(200)
np.random.shuffle(data_train)
np.random.seed(200)
np.random.shuffle(label_train)
np.random.seed(200)
np.random.shuffle(data_vali)
np.random.seed(200)
np.random.shuffle(label_vali)
np.random.seed(42)

# Creating an Instance history
save_dir = '/data/data2/jyx/chuankong/deep_0701/3dresnet10_230524_1'
history0 = LossHistory()
EarlyStop = EarlyStopping(monitor='val_loss', patience=20,verbose=1, mode='auto')
Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0005, cooldown=0, min_lr=1e-8)
checkpoint = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'),verbose=1, save_weights_only=False, period=2)
callbacks_list = [checkpoint]
# tbCallBack = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)

# model fit
model = Resnet3DBuilder.build_resnet_10((10,32,32, 1), 1)
model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])  # 'accuracy',
history = model.fit(data_train, label_train, validation_data=(X_test, y_test), epochs=num_epoch,batch_size=batch_size, class_weight = class_weight,callbacks=[history0,Reduce,callbacks_list])  # ,EarlyStop])#,saveBestModel])#,tbCallBack])#,EarlyStop
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Writing Data into .csv file
prediction_data = pd.DataFrame(model.predict(data_vali))
prediction_data.to_csv('/home/jyx/biyanai/deep_230524_1/test_prediction.csv')
prediction_data = pd.DataFrame(model.predict(data_train))
prediction_data.to_csv('/home/jyx/biyanai/deep_230524_1/train_prediction.csv')


fpr, tpr, thres = metrics.roc_curve(y_test, np.squeeze(model.predict(X_test)), pos_label=1)

optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thres)
print(optimal_th, optimal_point)
optimal_th = 0.1
print('-------------------------------------Prediction X_train')

X_train_auc, X_train_acc, X_train_sen = model_evalu(model, X_train, y_train, optimal_th)

print('-------------------------------------Prediction X_test')
X_test_auc, X_test_acc, X_test_sen = model_evalu(model, X_test, y_test, optimal_th)

# Prediction validation
print('####################################Prediction test')
test_auc,test_acc,test_sen = model_evalu(model, data_vali, label_vali, optimal_th)
print('   set             AUC              ACC             SENSITIVITY       \n',
      'X_train:',X_train_auc, X_train_acc, X_train_sen,'\n',
      'X_test:',X_test_auc, X_test_acc, X_test_sen,'\n',
      'test:',test_auc,test_acc,test_sen)
plot_roc(model,X_train,X_test,data_vali,y_train,y_test,label_vali,'last epoch')
#loss图
# history0.loss_plot('epoch')
# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/jyx/biyanai/deep_230524_1/accuracy.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/jyx/biyanai/deep_230524_1/loss.png')
plt.show()

# Save the model
model.save('/home/jyx/biyanai/deep_230524_1/my_model.h5')
json_string = model.to_json()
open('/home/jyx/biyanai/deep_230524_1/model_architecture_1.json','w').write(json_string)
yaml_string = model.to_yaml()
# open('/home/jyx/biyanai/deep_230328/model_architecture_2.yaml','w').write(yaml_string)
# model.save_weights('/home/jyx/biyanai/deep_230328/my_model_weights.h5')
#
###--------------------------------------------

# model1 = tf.keras.models.load_model('/data_raid5_21T/lyw/chuankong/deep_0701/3dresnet10_0805/model_144-0.06-0.49.hdf5')
# print(model1.predict(X_train))
# print(X_train.shape)
# save_dir = '/data_raid5_21T/lyw/chuankong/deep_0701/3dresnet10_0923'
# for num in np.linspace(2,200,100):
#     i="%02d" % (num)
#     for file in os.listdir(save_dir):
#         if '_'+i+'-' in file:
#             path = os.path.join(save_dir,file)
# model1 = tf.keras.models.load_model('/data_raid5_21T/lyw/chuankong/deep_0701/my_model.h5')
# # model1 = model.load_weights('/data_raid5_21T/lyw/chuankong/deep_0701/my_model_weights.h5')
#
# # fpr, tpr, thres = metrics.roc_curve(y_test, np.squeeze(model.predict(X_test)), pos_label=1)
# # optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thres)
# optimal_th = 0.2
# # print('-------------------------------------Prediction X_train')
# X_train_auc, X_train_acc, X_train_sen = model_evalu(model1, X_train, y_train, optimal_th)
# # Prediction train
# # print('-------------------------------------Prediction X_test')
# X_test_auc, X_test_acc, X_test_sen = model_evalu(model1, X_test, y_test, optimal_th)
# # Prediction validation
# # print('#####################################Prediction test')
# test_auc,test_acc,test_sen = model_evalu(model1, data_vali, label_vali, optimal_th)
# print('   set             AUC              ACC             SEN       \n',
#       'X_train:',X_train_auc, X_train_acc, X_train_sen,'\n',
#       'X_test:',X_test_auc, X_test_acc, X_test_sen,'\n',
#       'test:',test_auc,test_acc,test_sen)
#       # if X_train_auc>0.75 and X_train_acc>0.75 and X_train_sen>0.75 and X_test_auc>0.75 and X_test_acc>0.75 and X_test_sen>0.75 and test_auc>0.75 and test_acc>0.75 and test_sen>0.75:
#     # print(X_test_auc, X_test_acc, X_test_sen,'\n','      ',test_auc,test_acc,test_sen)
# plot_roc(model1, X_train, X_test, data_vali, y_train, y_test, label_vali,'my_model_weights')
