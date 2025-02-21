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

datapath='Data_Cortex_Nuclear.xls'
       
data_df=pd.read_excel(datapath)

_,_,labels=transform_and_return_data(data_df)      

    
from scipy.spatial.distance import pdist, squareform

cols = [col for col in data_df.columns if col not in ['Genotype', 'Treatment', 'Behavior', 'Behavior','class']]
short_cols = [col[0:77] for col in cols]
short_cols = [short_cols[i] + str(i) for i in range(1,len(short_cols),1)]

data_dist = pdist(data_df[cols].as_matrix().transpose())
print(data_dist)
data = Data([
    Heatmap(
        z=squareform(data_dist), colorscale='YIGnBu',
        x=short_cols,
        y=short_cols,     # y-axis labels
    )
])

layout = Layout(
    title='Transcription profiling of proteins',
    autosize=False,
    margin=Margin(
        l=200,
        b=200,
        pad=4
    ),
    xaxis=XAxis(
        showgrid=False, # remove grid
        autotick=False, # custom ticks
        dtick=1,        # show 1 tick per day
    ),
    yaxis=YAxis(
        showgrid=False,   # remove grid
        autotick=False,   # custom ticks
        dtick=1           # show 1 tick per day
    ),
)
fig = Figure(data=data, layout=layout)
py.iplot(fig, width=900, height=900)
