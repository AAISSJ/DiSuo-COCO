import pandas as pd

df=pd.read_csv('tong.csv',names=["과","이름","도","시","동"],encoding='cp949')

df.to_csv("tong_.csv",encoding='cp949',index=None)
