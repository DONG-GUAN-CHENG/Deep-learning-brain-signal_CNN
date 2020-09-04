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
###------------------------轉換panda為numpy數列
import numpy as np
xb1=np.array(x__train)
yb1=np.array(y__train)
yb2=np.array(y__trainsigch)
yb11=np.array(y__trainsigch11)
thbraw=np.array(thbraw)
##_----------------一行數列分割為一筆多少data
# (1) 測試集利用C20取4得到4845筆受試者資料，每筆受試者資料52個channel又可視為一筆，可得251940分為兩類別資料取1min和10min得到503880筆
# (1) 測試集利用C10取4得到210筆受試者資料，每筆受試者資料52個channel又可視為一筆，可得1pip920分為兩類別資料取1min和10min得到21840筆
x_train3D=xb1.reshape(78064,128,1).astype('float64') #66248 22484 71876 74256
y_train3D=yb1.reshape(578,128,1).astype('float64')
y_train3D2=yb2.reshape(4148,128,1).astype('float64')
y_train3D3=yb11.reshape(3060,128,1).astype('float64')
ythb=thbraw.reshape(204,128,1).astype('float64')
## ---------------------------匯入創建神經網路所需資料集，利用keras 及 sklearn------------------------
from keras.models import Sequential #匯入資料集
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D,AveragePooling1D,BatchNormalization,Activation #匯入Keras的layer模組
from keras.layers.convolutional import Conv1D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from numpy import array, argmax
print("\n--- Create neural network model ---\n")
## ------------------首先進行資料的預處理對Train和Test data ，會產生feature 和 Label值-----------------
#feature 特徵值匯入
x_train3dD= x_train3D.reshape(x_train3D.shape[0],128,1).astype('float64') #將features 特徵值 轉換為三維矩陣
y_train3dD=y_train3D.reshape(y_train3D.shape[0],128,1).astype('float64')
y_train3dD2=y_train3D2.reshape(y_train3D2.shape[0],128,1).astype('float64')
y_train3dD3=y_train3D3.reshape(y_train3D3.shape[0],128,1).astype('float64')
#label  真實的label匯入
#(1) 先進行label encoding，由於這本質上是類別資料，並沒有順序大小之分，故後續還需進行一次one hot encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_le_trainx=pd.DataFrame(x_label)
data_le_trainy=pd.DataFrame(y_label)
data_le_trainy2=pd.DataFrame(y_labelsigch)
data_le_trainy3=pd.DataFrame(y_labelsigch11)
data_le_trainx['label'] = labelencoder.fit_transform(data_le_trainx['label'])
data_le_trainy['label'] = labelencoder.fit_transform(data_le_trainy['label'])
data_le_trainy2['label'] = labelencoder.fit_transform(data_le_trainy2['label'])
data_le_trainy3['label'] = labelencoder.fit_transform(data_le_trainy3['label'])
#(2) 在Label Encoder之後，我們可能會混淆我們的機器學習模型，認為列具有某種順序或層次結構的數據。為避免這種情況，我們將使用'OneHotEncode'
from sklearn.preprocessing import OneHotEncoder
x_labelonehot=np_utils.to_categorical(data_le_trainx)
y_labelonehot=np_utils.to_categorical(data_le_trainy)
y_labelonehot2=np_utils.to_categorical(data_le_trainy2)
y_labelonehot3=np_utils.to_categorical(data_le_trainy3)
##-------------打亂train 資料-------------
from sklearn.utils import shuffle
x_train3dD,x_labelonehot=shuffle(x_train3dD,x_labelonehot,random_state=0) #1,0 (0)->female 0,1 (1)->male
y_train3dD,y_labelonehot=shuffle(y_train3dD,y_labelonehot,random_state=0)
y_train3dD2,y_labelonehot2=shuffle(y_train3dD2,y_labelonehot2,random_state=0)
y_train3dD3,y_labelonehot3=shuffle(y_train3dD3,y_labelonehot3,random_state=0)
##-------------------建立線性推疊模型，後續只要利用model.add()方法，將神經網路層加入模型-------------------
# -----(1) 建立卷積層 1 池化層 1
model= Sequential()
model.add(Conv1D(filters=8,kernel_size=(11), activation='relu',strides = 1,padding='SAME',input_shape=(128,1),name='conv1D_1'))
#model.add(BatchNormalization())
#model.add(MaxPooling1D(2))
#prediction=model.predict_classes(x_train3dD[:1,:])
#import matplotlib.pyplot as plt
#train_img=np.reshape(x_train3dD[:1,:],(128,1))
#plt.matshow(train_img,cmap=plt.get_cmap('binary'))
#plt.show()
#import matplotlib.pyplot as plt
#cov_img=np.reshape(prediction[:1,:],(64,1))
#plt.matshow(cov_img,cmap=plt.get_cmap('binary'))
#plt.show()
model.add(Conv1D(filters=16,kernel_size=(9), activation='relu',strides = 1,padding='SAME',input_shape=(128,1),name='conv1D_2'))
model.add(MaxPooling1D(2,strides=1,name='Max_Pooling1D_3',padding='SAME'))
model.add(Conv1D(filters=32,kernel_size=(7), activation='relu',strides = 1,padding='SAME',input_shape=(128,1),name='conv1D_4'))
#model.add(MaxPooling1D(2))
model.add(Conv1D(filters=48,kernel_size=(5), activation='relu',strides = 1,padding='SAME',input_shape=(128,1),name='conv1D_5'))
#model.add(Dropout(0.5))
#model.add(MaxPooling1D(2))
#model.add(Conv1D(filters=64,kernel_size=(5), activation='relu',strides = 1,padding='SAME',input_shape=(128,1)))
#model.add(AveragePooling1D(2))
#model.add(MaxPooling1D(2))
#model.add(Dropout(0.5))
model.add(Flatten(name='Flatten_6'))
#model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5,name='Dropout_7')) #避免overfitting
model.add(Dense(2, activation='sigmoid',name='Dense_8'))
print(model.summary())
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)
##---------------------------
#Back Propagation 反向傳播算法
from keras.optimizers import SGD
from keras import optimizers
print("\n--- Fit the model ---\n")
#------------------------定義訓練方式---------------
#adam=optimizers.Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])# optimizer='adam'
              #categorical_crossentropy設定損失函數，在deep learning中使用交叉殤，訓練效果比較好#最優化方法，可以讓訓練更快收斂，提高準確率

# Hyper-parameters
BATCH_SIZE = 64  #每一批次幾筆資料
EPOCHS =150 #訓練週期


#from keras.callbacks import EarlyStopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
#early_stopping = EarlyStopping(monitor='val_train', patience=50, verbose=2)
# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.

from time import perf_counter
t1_start = perf_counter()
train_history = model.fit(x_train3dD,x_labelonehot,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    #callbacks=[early_stopping],
                    validation_split=0.2,
                    verbose=1)
                      #80%,作為訓練資料，20%作為驗證資料
t1_stop = perf_counter()

print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in seconds:",
      t1_stop - t1_start)


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

show_train_history(train_history,'accuracy','val_accuracy')
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
prediction=model.predict_classes(y_train3dD)
prediction[:10]
scores=model.evaluate(y_train3dD2, y_labelonehot2, verbose=1,batch_size=64)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])
scores=model.evaluate(y_train3dD3, y_labelonehot3, verbose=1,batch_size=64)
print()
print('Test loss=',scores[0])
print('Test accuracy=',scores[1])
prediction2=model.predict_classes(y_train3dD2)
prediction2[:10]

prediction=model.predict_classes(y_train3dD)
prediction_probability=model.predict(y_train3dD)
prediction2=model.predict_classes(y_train3dD2)
prediction_probability2=model.predict(y_train3dD2)
prediction3=model.predict_classes(y_train3dD3)
prediction_probability3=model.predict(y_train3dD3)
## SVM
x_train3dDd=x_train3dD.reshape(78064,128)
x_testlabeld=np.delete(x_labelonehot,0,axis=1) #預測資料基礎標籤值
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf',probability=True,gamma='auto')
svclassifier.fit(x_train3dDd,x_testlabeld)
predicted=svclassifier.predict(y_train3dD2)
predicted_prob=svclassifier.predict_proba(y_train3dD2)
##
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = np.mean(scores), np.std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
##
# run an experiment
def run_experiment(repeats=1):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = model.evaluate(y_train3dD2, y_labelonehot2)
        accsc=score[1]
		accvalue = accsc * 100.0
		print('>#%d: %.3f' % (r+1, accvalue))
		scores.append(accvalue)
	# summarize results


## 儲存模型
import tensorflow as tf
model.save('1D CNN frequency_ALFF4layer 11 epochs150.h5')
import tensorflow as tf
model = tf.contrib.keras.models.load_model('1D CNN frequency_ALFF4layer 11 epochs150_89.13.h5')
## -------- Confusion matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
arr=np.delete(y_labelonehot2,0,axis=1) #轉換test data真實值為一維
pd.crosstab(arr.reshape(-1),prediction2,rownames=['label'],colnames=['predict'])


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
cnf_matrix = confusion_matrix(arr.reshape(-1), prediction2)
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,title='confusion matrix')   #title="month = " + str(month)
plt.show()
## 繪製ROC曲線----------------------------------------------------------------
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
y_pred_keras=np.delete(prediction_probability2,1,axis=1) #預測資料機率值 # prediction_probability2, y_labelonehot2
y_testlabel=np.delete(y_labelonehot2,1,axis=1) #預測資料基礎標籤值
#np.save("filename.npy",prediction_probability2)
#b = np.load("filename.npy")
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
#y_score = classifier.fit(x_train3dD,x_labelonehot).decision_function(y_train3dD)
fpr1, tpr1, thresholds = roc_curve(y_testlabel,y_pred_keras,pos_label=1) #測試資料所得的label 後面的為測試資料預測機率
roc_auc1 = auc(fpr1, tpr1)
#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
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
precision_recall_fscore_support(y_truelabel, prediction2, average='weighted')
## 精確率(Precision), 召回率Recall, F1-score 計算
from sklearn.metrics import classification_report
target_names = ['female', 'male']
cr =classification_report(y_truelabel, prediction2,target_names=target_names,digits=5)
print(classification_report(y_truelabel, prediction2,target_names=target_names,digits=5))
#plot_classification_report(cr)
lines = cr.split('\n')  #刪除report內為0的值，或有空格的地方
del  lines[1]
del  lines[3]
del  lines[3]
del  lines[5]
lines[3]='macro_avg 0.89192 0.88993 0.89076 4148'
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
plot_model(model, to_file='model.png',show_shapes=True)



