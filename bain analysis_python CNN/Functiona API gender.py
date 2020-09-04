import tensorflow as tf
#tf.compat.v1.Session()
## ------首先匯入.csv檔的資料-------
import pandas as pd
x__train = pd.read_csv('x_train 34 channelnew4.csv') #前female後male  x_traingennew  x_labelgennew->66248  x_train 34 channel.csv, x_label 34 channel.csv->69496
x_label = pd.read_csv('x_label 34 channelnew4.csv') #a ->0->1,0  b->1->0,1 x_trainsigchannel  x_trainsigchannel_label->22484
y__train = pd.read_csv('xraw_train 34 channel.csv')
y_label = pd.read_csv('xraw_label 34 channel.csv')
y__trainsigch = pd.read_csv('y_test 34 channelnew3.csv') #y_test 34 channel y_testlabel 34 channel->4352  #y_test 34 channelnew3 y_testlabel 34 channelnew3->4148
y_labelsigch = pd.read_csv('y_testlabel 34 channelnew3.csv')
y__trainsigch11 = pd.read_csv('y_trainfi.csv')
y_labelsigch11 = pd.read_csv('y_labelfi.csv')
thbraw = pd.read_csv('thbfe 6p34 channel.csv')
y__trainthb = pd.read_csv('thb raw data 34 channel.csv')
y_labelsigthb = pd.read_csv('thb raw data 34 channel label.csv')
###------------------------轉換panda為numpy數列
import numpy as np
xb1=np.array(x__train)
yb1=np.array(y__train)
yb2=np.array(y__trainsigch)
yb11=np.array(y__trainsigch11)
thbraw=np.array(thbraw)
ybthb=np.array(y__trainthb)
##_----------------一行數列分割為一筆多少data
# (1) 測試集利用C20取4得到4845筆受試者資料，每筆受試者資料52個channel又可視為一筆，可得251940分為兩類別資料取1min和10min得到503880筆
# (1) 測試集利用C10取4得到210筆受試者資料，每筆受試者資料52個channel又可視為一筆，可得1pip920分為兩類別資料取1min和10min得到21840筆
x_train3D=xb1.reshape(78064,128,1).astype('float64') #66248 22484 71876 74256
y_train3D=yb1.reshape(578,128,1).astype('float64')
y_train3D2=yb2.reshape(4148,128,1).astype('float64')
y_train3D3=yb11.reshape(3060,128,1).astype('float64')
ythb=thbraw.reshape(204,128,1).astype('float64')
y_trainthb=ybthb.reshape(1258,128,1).astype('float64') #thb raw data test
## ---------------------------匯入創建神經網路所需資料集，利用keras 及 sklearn------------------------
from keras.models import Sequential #匯入資料集
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D,AveragePooling1D,BatchNormalization,Activation #匯入Keras的layer模組
from keras.layers.convolutional import Conv1D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from numpy import array, argmax
from datetime import datetime
from packaging import version
import tensorflow as tf
from tensorflow import keras
import tensorboard
print("\n--- Create neural network model ---\n")
## ------------------首先進行資料的預處理對Train和Test data ，會產生feature 和 Label值-----------------
#feature 特徵值匯入
x_train3dD= x_train3D.reshape(x_train3D.shape[0],128,1).astype('float64') #將features 特徵值 轉換為三維矩陣
y_train3dD=y_train3D.reshape(y_train3D.shape[0],128,1).astype('float64')
y_train3dD2=y_train3D2.reshape(y_train3D2.shape[0],128,1).astype('float64')
y_train3dD3=y_train3D3.reshape(y_train3D3.shape[0],128,1).astype('float64')
y_trainthb=y_trainthb.reshape(y_trainthb.shape[0],128,1).astype('float64')
#label  真實的label匯入
#(1) 先進行label encoding，由於這本質上是類別資料，並沒有順序大小之分，故後續還需進行一次one hot encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_le_trainx=pd.DataFrame(x_label)
data_le_trainy=pd.DataFrame(y_label)
data_le_trainy2=pd.DataFrame(y_labelsigch)
data_le_trainy3=pd.DataFrame(y_labelsigch11)
data_le_trainythb=pd.DataFrame(y_labelsigthb)
data_le_trainx['label'] = labelencoder.fit_transform(data_le_trainx['label'])
data_le_trainy['label'] = labelencoder.fit_transform(data_le_trainy['label'])
data_le_trainy2['label'] = labelencoder.fit_transform(data_le_trainy2['label'])
data_le_trainy3['label'] = labelencoder.fit_transform(data_le_trainy3['label'])
data_le_trainythb['label'] = labelencoder.fit_transform(data_le_trainythb['label'])
#(2) 在Label Encoder之後，我們可能會混淆我們的機器學習模型，認為列具有某種順序或層次結構的數據。為避免這種情況，我們將使用'OneHotEncode'
from sklearn.preprocessing import OneHotEncoder
x_labelonehot=np_utils.to_categorical(data_le_trainx)
y_labelonehot=np_utils.to_categorical(data_le_trainy)
y_labelonehot2=np_utils.to_categorical(data_le_trainy2)
y_labelonehot3=np_utils.to_categorical(data_le_trainy3)
y_labelonehotthb=np_utils.to_categorical(data_le_trainythb)
##-------------打亂train 資料-------------
from sklearn.utils import shuffle
x_train3dD,x_labelonehot=shuffle(x_train3dD,x_labelonehot,random_state=0) #1,0 (0)->female 0,1 (1)->male
y_train3dD,y_labelonehot=shuffle(y_train3dD,y_labelonehot,random_state=0)
y_train3dD2,y_labelonehot2=shuffle(y_train3dD2,y_labelonehot2,random_state=0)
y_train3dD3,y_labelonehot3=shuffle(y_train3dD3,y_labelonehot3,random_state=0)
y_trainthb,y_labelonehotthb=shuffle(y_trainthb,y_labelonehotthb,random_state=0)
## Functional Model API (Advanced) 模型的建構
inputs = tf.keras.Input(shape=(128,1))
x = tf.keras.layers.Conv1D(8, kernel_size=(11), activation='relu',strides=1,padding='SAME')(inputs)
x = tf.keras.layers.Conv1D(16, kernel_size=(9), activation='relu',strides=1,padding='SAME')(x)
x = tf.keras.layers.MaxPooling1D(pool_size=(2),strides=1,padding='SAME')(x)
x = tf.keras.layers.Conv1D(32, kernel_size=(7), activation='relu',strides=1,padding='SAME')(x)
x = tf.keras.layers.Conv1D(48, kernel_size=(5), activation='relu',strides=1,padding='SAME')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
#plot_model(model,to_file='ff.png')
#logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#def scheduler(epoch):
 # if epoch < 10:
    #return 0.001
 # else:
    #return 0.001 * tf.math.exp(0.1 * (10 - epoch))
callbacks_list = [
    #tf.keras.callbacks.ModelCheckpoint(
       # filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
       # monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10),
   # tf.keras.callbacks.LearningRateScheduler(scheduler),
    #tf.keras.callbacks.TensorBoard(log_dir="./logs")
]
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tf.keras.losses.binary_crossentropy,metrics = ['accuracy'])
    #metrics=[tf.keras.metrics.sparse_categorical_accuracy]

train_history=model.fit(x_train3dD,x_labelonehot, epochs=200, batch_size=64, validation_split=0.2,verbose=1,callbacks=callbacks_list)

## ------------畫出accuracy和loss執行結果曲線---------------
#全域變數設為time new roman
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


print("\n--- Learning curve of model training ---\n")
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.figure(num=3, figsize=(7, 5))
    plt.plot(train_history.history[train],linewidth=3.0)
    plt.plot(train_history.history[validation],linewidth=3.0)
    plt.title('train history',fontsize=16,fontweight='bold')
    plt.ylabel(train,fontsize=16,fontweight='bold')
    plt.xlabel('epoch',fontsize=16,fontweight='bold')
    plt.legend(['train','validation'],loc='best',fontsize=16)
    plt.xticks(fontsize=16,fontweight='bold')
    plt.yticks(fontsize=16,fontweight='bold')
    plt.grid(True)
    ax = plt.gca()
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    plt.show()
    # 设置刻度字体大小

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
## -----------------------評估模型準確率-------------
scores=model.evaluate(x_train3dD, x_labelonehot, verbose=0)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])

scores=model.evaluate(y_train3dD, y_labelonehot, verbose=1,batch_size=64)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])

scores=model.evaluate(y_train3dD2, y_labelonehot2, verbose=1,batch_size=64)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])
scores=model.evaluate(y_train3dD3, y_labelonehot3, verbose=1,batch_size=64)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])

scores=model.evaluate(y_trainthb,y_labelonehotthb, verbose=1,batch_size=64)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])

prediction=model.predict(y_train3dD)    #因為是函數建模所以要用組合形式預測 (機率)
predictionCC=np.argmax(prediction,axis=1)     #argmax為返回最大數的索引值 (Class)
prediction2=model.predict(y_train3dD2)
prediction2CC=np.argmax(prediction2,axis=1)
prediction3=model.predict(y_train3dD3)
prediction3CC=np.argmax(prediction3,axis=1)
## 儲存模型
import tensorflow as tf
model.save('1D CNN frequency_ALFF4layer 11 epochs150.h5')
import tensorflow as tf
model = tf.contrib.keras.models.load_model('1D CNN frequency_ALFF4layer 11 epochs150_89.13.h5')
## -------- Confusion matrix 混淆矩陣
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
arr=np.delete(y_labelonehot2,0,axis=1) #轉換test data真實值為一維
pd.crosstab(arr.reshape(-1),prediction2CC,rownames=['label'],colnames=['predict'])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=16)
    plt.grid(False)
    plt.title("Confusion matrix", fontsize=16)
    plt.ylabel('True label',fontsize=16)
    plt.xlabel('Predicted label',fontsize=16)
    plt.tight_layout()

month='3'
target_names = ['female','male']
plt.figure(num=1)
cnf_matrix = confusion_matrix(arr.reshape(-1), prediction2CC)
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,title='confusion matrix')   #title="month = " + str(month)
plt.show()
## ----------------------------------繪製ROC曲線--------------------------------------
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
y_pred_keras=np.delete(prediction2,1,axis=1) #預測資料機率值 # prediction_probability2, y_labelonehot2
y_testlabel=np.delete(y_labelonehot2,1,axis=1) #預測資料基礎標籤值
fpr1, tpr1, thresholds = roc_curve(y_testlabel,y_pred_keras,pos_label=1) #測試資料所得的label 後面的為測試資料預測機率
roc_auc1 = auc(fpr1, tpr1)
#畫圖，只需要plt.plot(fpr,tpr)
plt.figure(num = 2, figsize = None)
plt.plot(fpr1, tpr1,color='darkorange', lw=2, label='ROC(area = %0.2f)' % (roc_auc1))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("FPR (False Positive Rate)",fontsize=16)
plt.ylabel("TPR (True Positive Rate)",fontsize=16)
plt.title("Receiver Operating Characteristic, ROC(AUC = %0.2f)"% (roc_auc1),fontsize=16)
plt.legend(loc="lower right",frameon=True,edgecolor='black',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(False)
ax = plt.gca()
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
plt.show()

## 對測試資料的label做預處理
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
y_truelabel=np.delete(y_labelonehot2,0,axis=1) #預測資料基礎標籤值
#prediction2 #預測資料基礎標籤值
precision_recall_fscore_support(y_truelabel, prediction2CC, average='weighted')
## 精確率(Precision), 召回率Recall, F1-score 計算
from sklearn.metrics import classification_report
target_names = ['female', 'male']
cr =classification_report(y_truelabel, prediction2CC,target_names=target_names,digits=5)
print(classification_report(y_truelabel, prediction2CC,target_names=target_names,digits=5))
#plot_classification_report(cr)
lines = cr.split('\n')  #刪除report內為0的值，或有空格的地方
del  lines[1]
del  lines[3]
del  lines[3]
del  lines[5]
lines[3]='macro_avg 0.89192 0.88993 0.89076 4148'     #需要進行更改因為是固定的
lines[4]='weighted_avg 0.89180 0.89176 0.89162 4148'
## 數值分析以熱力圖表示
from matplotlib import colors
from matplotlib.colors import ListedColormap
ddl_heat = ['#DBDBDB','#DCD5CC','#DCCEBE','#DDC8AF','#DEC2A0','#DEBB91',\
            '#DFB583','#DFAE74','#E0A865','#E1A256','#E19B48','#E29539']
ddlheatmap = colors.ListedColormap(ddl_heat)

def plot_classification_report(cr, title=None, cmap=ddlheatmap):
    title = title or 'Classification report'
    classes = []
    matrix = []
    for line in lines[1:(len(lines)-1)]: #對lines的索引做取值
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]] #line分割後再把裡面的元素在做一次分割
        matrix.append(value)
    fig, ax = plt.subplots(1,num=3)
    for column in range(len(matrix)):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center',fontsize=20)
    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=20)
    plt.clim(0.5,1)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=10,fontsize=20)
    plt.yticks(y_tick_marks, classes,fontsize=20)
    plt.ylabel('Classes',fontsize=20)
    plt.xlabel('Measures',fontsize=20)
    plt.grid()
    plt.show()

 plot_classification_report(cr)

##可視化網路結構
from keras.utils import plot_model
tf.keras.utils.plot_model(model, to_file='model.png',show_shapes=True)