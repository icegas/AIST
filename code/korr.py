import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

df=pd.read_csv('/root/Desktop/AISTHack/code/train/chats_train.csv', index_col=False)
df=df.dropna(how='any')
df=df[df['text'].str.lower().str.contains("trx|thron", regex=True)]
post_per_sec=df['timestamp'].value_counts().reset_index()

post_per_sec = post_per_sec.sort_values(by=['index'])
timestemps = np.array(post_per_sec['index'].values)
words = np.array(post_per_sec['timestamp'].values)

#datetime.datetime.fromtimestamp(int(

j, values_2_hours = 0, []
##print(post_per_sec)
tmp = []
#print(timestemps)


for i in range(len(timestemps)):
    if (timestemps[i] - timestemps[j]) <= 7200:
        tmp.append(words[i])
    else:
        inter = int( (timestemps[i] - timestemps[j]) / 7200)

        if(inter > 1):
            for _ in range(inter):
                values_2_hours.append(0)


        values_2_hours.append(np.sum(tmp))
        tmp=[]
        j = i


values_2_hours.append(np.sum(tmp))
size = 668

dataFile = '/root/Desktop/AISTHack/code/train/tickers_train.csv'
data = pd.read_csv(dataFile)
temp = data['name']
crypto = 'TRX'#"NANO" 
TRX = np.array([])
#print(data['priceBtc'].value)
data = pd.DataFrame(data)

data = data[data['ticker'].str.contains(crypto)]

for i in range(len(data)):
    TRX = np.append(TRX, data['priceBtc'].iloc[i])

y = []
length = int(len(TRX) - size)
x = np.array([i for i in range(length)])
for i in range(length):
    y.append(np.corrcoef(TRX[i : size + i], values_2_hours[: size]))
yy = np.array([y[i][0, 1] for i in range(length) ])
plt.plot(x, yy)
plt.show()



