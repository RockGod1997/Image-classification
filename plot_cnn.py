import pandas as pd

import matplotlib.pyplot as plt    
data = pd.read_csv("results_cnn_2.csv",delimiter=',')
data1 = pd.read_csv("results_gru_2.csv",delimiter=',')
df=pd.concat([data,data1])
df.plot(x='Train', y=['CNN_F','GRU_F','LSTM_F'])
plt.title(" CLASS CLASSIFICATION")
plt.ylabel('F1 Scores')
plt.show()
