from read_data import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
import os

figures_path='figures'

if not os.path.exists(figures_path):
    os.mkdir(figures_path)


datapath='heart_failure_clinical_records_dataset.csv'

        
data_df=read_file(datapath)

_,_,labels=transform_and_return_data(data_df)      

    
pp = sns.pairplot(data_df[labels], size=1.8, aspect=1.8,
          plot_kws=dict(edgecolor="k", linewidth=0.5),
          diag_kind="kde", diag_kws=dict(shade=True))


fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Features Pairwise Plots', fontsize=14)


fig.savefig(figures_path+os.path.sep+"output.pdf")

