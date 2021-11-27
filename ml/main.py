import numpy as np
from sklearn import preprocessing
input_data = np.array([[2.1, -1.9, 5.5],
                      [-1.5, 2.4, 3.5],
                      [0.5, -7.9, 5.6],
                      [5.9, 2.3, -5.8]])
# Data Preprocessing
    #binarization
data_binarized = preprocessing.Binarizer(threshold=0.5).transform(input_data)
print(data_binarized)

    #mean removal
Mean_data = input_data.mean(axis=0)
Std_data = input_data.std(axis=0)
print(Mean_data,"\n\n",Std_data,"\n\n")
    #removing the mean and std from the input data
data_scaled = preprocessing.scale(input_data)
mean_data = data_scaled.mean(axis=0)
std_data = data_scaled.std(axis=0)
print(data_scaled,"\n\n",mean_data,"\n\n",std_data)

    #scaling
        #min max scaling
data_scaler_minmaxa = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaler_minmaxb = data_scaler_minmaxa.fit_transform(input_data)
print("data_scaler_minmaxa: ",data_scaler_minmaxa, "\n\n", "data_scaler_minmaxb: ", data_scaler_minmaxb)

    #Normalization
        #L1 normalization
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("data_normalized_l1: ", data_normalized_l1, "\n\n", "data_normalized_l2: ", data_normalized_l2)



