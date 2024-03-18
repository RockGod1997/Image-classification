import pandas as pd

import matplotlib.pyplot as plt    
data = pd.read_csv("results_2_50k.csv",delimiter=',')


data.plot(x='Train', y=['LR_F','NB_F','SVM_F'])
plt.title("Binary CLASS CLASSIFICATION")
plt.ylabel('F1 Scores')
plt.show()
