## 使用t-SNE可視化數據
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
##-------------------------------獲取模型最後一層的數據--------------------------------
# 获取x = tf.keras.layers.Flatten()(x)數據
def create_truncated_model(trained_model):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=(11), activation='relu', strides=1, padding='SAME', input_shape=(128, 1),name='conv1D_1'))
    model.add(Conv1D(filters=16, kernel_size=(9), activation='relu', strides=1, padding='SAME', input_shape=(128, 1),name='conv1D_2'))
    model.add(MaxPooling1D(2, strides=1, name='Max_Pooling1D_3', padding='SAME'))
    model.add(Conv1D(filters=32, kernel_size=(7), activation='relu', strides=1, padding='SAME', input_shape=(128, 1),name='conv1D_4'))
    model.add(Conv1D(filters=48, kernel_size=(5), activation='relu', strides=1, padding='SAME', input_shape=(128, 1),name='conv1D_5'))
    #model.add(Conv1D(filters=32, kernel_size=(7), activation='relu', strides=1, padding='SAME', input_shape=(128, 1),name='conv1D_4'))
   # model.add(Conv1D(filters=48, kernel_size=(5), activation='relu', strides=1, padding='SAME', input_shape=(128, 1),name='conv1D_5'))
    #inputs = tf.keras.Input(shape=(128, 1))
    #x = tf.keras.layers.Conv1D(8, kernel_size=(11), activation='relu', strides=1, padding='SAME')(inputs)
    #x = tf.keras.layers.Conv1D(16, kernel_size=(9), activation='relu', strides=1, padding='SAME')(x)
    #x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=2, padding='SAME')(x)
   # x = tf.keras.layers.Conv1D(32, kernel_size=(7), activation='relu', strides=1, padding='SAME')(x)
    #x = tf.keras.layers.Conv1D(48, kernel_size=(5), activation='relu', strides=1, padding='SAME')(x)
    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.Dense(units=2, activation='sigmoid')(x)
    model.add(Flatten(name='Flatten_6'))
    model.add(Dropout(0.5, name='Dropout_7'))  # 避免overfitting
    model.add(Dense(2, activation='sigmoid', name='Dense_8'))
    for i, layer in enumerate(model.layers):
        layer.set_weights(trained_model.layers[i].get_weights())
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model
arr=np.delete(y_labelonehotthb,0,axis=1) #轉換test data真實值為一維
truncated_model = create_truncated_model(model)
hidden_features = truncated_model.predict(y_trainthb)

#book_em=model.get_layer('conv1D_1')
#book_em_weights=book_em.get_weights()[0]
##-------------------------------PCA,tSNE降维分析--------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components=2)# 总的类别
pca_result = pca.fit_transform(hidden_features)
print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
import seaborn as sns
#Run T-SNE on the PCA features.
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose = 2,perplexity=15,n_iter=2000)
tsne_results = tsne.fit_transform(pca_result[:4148])
sns.scatterplot(tsne_results[:,0],tsne_results[:,1])
#-------------------------------可视化--------------------------------
num_classes= ['female', 'male']
y_test_cat = np_utils.to_categorical(arr[:4148], num_classes = 2)# 总的类别
color_map = np.argmax(y_test_cat, axis=1)
plt.figure(figsize=(None))
for cl in range(2):# 总的类别
    indices = np.where(color_map==cl)
    indices = indices[0]
    female=tsne_results[indices,0]
    male= tsne_results[indices, 1]
    plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl,s=10)
    plt.title("Visualization_conv1D_1", fontsize=16)
    plt.ylabel('2nd dimension',fontsize=16)
    plt.xlabel('1st dimension',fontsize=16)
    plt.legend(loc="best", frameon=True, edgecolor='black', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

##t-SNE
y_train3dD3=y_train3D3.reshape(3060,128).astype('float64')
X_tsne = manifold.TSNE(n_components=2, init='pca', random_state=5, verbose=1).fit_transform(y_train3dD3)

#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne-x_min) / (x_max-x_min)  #Normalize
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0],X_norm[i, 1], str(arr[i]), color=plt.cm.Set1(arr[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()


import pandas
arr=np.delete(y_labelonehot3,0,axis=1) #轉換test data真實值為一維
arr=arr.reshape(3060,).astype('float64')
y_train3dDraw=y_train3Draw.reshape(3848,128).astype('float64')

df = pandas.DataFrame(dict(Feature_1=X_tsne[:,0], Feature_2=X_tsne[:,1], label=arr))

df.plot(x="Feature_1", y="Feature_2", kind='scatter', c='label', colormap='viridis')


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
X_embedded = TSNE(n_components=2).fit_transform(y_trainthb)

plt.figure()
plt.scatter(X_embedded[:,0],X_embedded[:,1],c=arr, s=0.5, alpha = 0.5)

plt.show()
