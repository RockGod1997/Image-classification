import pandas as pd

import matplotlib.pyplot as plt    
data = pd.read_csv("results_2_50k.csv",delimiter=',')
data1 = pd.read_csv("results_cnn_2_50k.csv",delimiter=',')
data2 = pd.read_csv("results_gru_2_50k.csv",delimiter=',')
data3 = pd.read_csv("results_lstm_2_50k.csv",delimiter=',')
df=pd.concat([data,data1,data2,data3])

df.plot(x='Train', y=['LR_F','NB_F','SVM_F','CNN_F','GRU_F','LSTM_F'])
plt.title(" Binary CLASS CLASSIFICATION")
plt.ylabel('F1 Scores')
plt.xlabel('Train Set Size')
plt.show()
