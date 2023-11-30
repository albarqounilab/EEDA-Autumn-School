- Load Libraries (5 min)
> - google and write a couple of words about what each library is usful for?
> - xample: Pandas: data manipulation and analysis \\
> - which line is a pachage which one is a funciton?


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



- Download and Load Dataset
> - google "boston.csv" and download the file \\
> - put the address of the file in the read_csv \\
> - get info about samples in the dataset (features as coloumns)
> - we want a clean table without extra feature columns


dataset = pd.read_csv('sample_data/Boston.csv', delimiter=r",")
# dataset.info()
dataset.head()


col_names = dataset.columns.to_list()
print(col_names)
dataset = dataset.drop(columns=["Unnamed: 0"])
col_names = dataset.columns.to_list()
print(col_names)
dataset.head()


- Plot Feature Distributions
> - google scatter and histogram
> - 196 plots (14 by 14) \\
> - each plot is one feature against another feature


# making a super plot with 14 by 14 subplot
fig, axs = plt.subplots(len(col_names), len(col_names), figsize=(25, 25))

# going throught the 14 by 14 subplots
for i in range(len(col_names)):
  for j in range(len(col_names)):
    # plot histogram if idential
    if i == j:
      axs[i,j].hist(dataset[col_names[i]])
    # plot the corelation if non-identical
    else:
      axs[i,j].scatter(dataset[col_names[i]], dataset[col_names[j]])
  #   axs[0,j].set_title(column_names[j])
  # axs[i,0].set(ylabel=column_names[i])

plt.show()
