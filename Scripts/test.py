df_series = pd.Series(np.random.normal(0,1,10000))
from scipy.stats import jarque_bera
#Getting skewness and excess jurtosis
s = df_series.skew()
ex_k = df_series.kurt()
jb = jarque_bera(df_series)
jb_stat = '{:.4f}'.format(float(jb[0]))
jb_prob = '{:.4f}'.format(float(jb[1]))

#Appending extra stats
y_desc = df_series.describe()
import pandas as pd
pd.options.display.float_format = '{:<10d}'.format
	
stats_plus = pd.Series([s,ex_k, jb_stat, jb_prob],index=["skewness", "excess kurtosis", "jarque-bera", "JB prob of normality"])   

y_desc = y_desc.append(stats_plus)
