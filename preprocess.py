import pandas as pd
import datetime
import numpy as np

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('data.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df = df[df['Date'] >= datetime.date(1999, 2, 8)].reset_index(drop=True)

df = df.drop(['Close'], axis=1)
print(df.describe())
print(df.shape)

print(df[df['Adj Close'] == 0])
# 1398 2004-08-31   0.0   0.0  0.0        0.0  163580000
# 2274 2008-02-25   0.0   0.0  0.0        0.0  288600000
# 4272 2016-02-01   0.0   0.0  0.0        0.0  114450000

print(df[df['Adj Close'] == 1000000])
# 440  2000-11-02   100000.0   1000000.0  1000000.0  1000000.0  250440000
# 3330 2012-05-02  1000000.0   1000000.0  1000000.0  1000000.0  100770000
# 4669 2017-08-28   800002.5  10000000.0  1000000.0  1000000.0  218740000

########## Replace wrong data
tmp_df = df[~df['Adj Close'].isin([0, 1000000])]

open_avg_diff = pd.Series.mean(tmp_df['Open'] - tmp_df['Adj Close'])
high_avg_diff = pd.Series.mean(tmp_df['High'] - tmp_df['Adj Close'])
low_avg_diff = pd.Series.mean(tmp_df['Low'] - tmp_df['Adj Close'])

# https://www.tradingview.com/chart/?symbol=DJ%3ADJI
df.loc[1398, 'Adj Close'] = 10173.92  # from official web
df.loc[1398, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[1398, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[1398, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[2274, 'Adj Close'] = 12570.22  # from official web
df.loc[2274, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[2274, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[2274, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[4272, 'Adj Close'] = 16449.18  # from official web
df.loc[4272, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[4272, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[4272, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[440, 'Adj Close'] = 10880.51  # from official web
df.loc[440, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[440, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[440, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[3330, 'Adj Close'] = 13268.57  # from official web
df.loc[3330, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[3330, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[3330, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[4669, 'Adj Close'] = 21808.4  # from official web
df.loc[4669, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[4669, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[4669, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

print(df.describe())
cont_features = [
    'Open', 'High',
    'Low', 'Adj Close',
    'Volume']

########### Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(df[cont_features])
print(scaled)

df[cont_features] = pd.DataFrame(scaled)

########### Compute Price Change
size = df.shape[0]
output_list = []
for i, row in df.iterrows():
    cur = row['Adj Close']
    if (i == 0):
        output_list.append(np.nan)
        prev = cur
        continue
    else:
        if cur > prev:
            output_list.append(1)
        else:
            output_list.append(0)

    prev = cur
#end def
df['PChange'] = pd.Series(output_list)

########### Get date info
df['Day of Week'] = df.apply(lambda x: x['Date'].weekday(), axis=1)
df['Day of Week'] = df['Day of Week'].astype('category')

df['Month of Year'] = df.apply(lambda x: x['Date'].month, axis=1)
df['Month of Year'] = df['Month of Year'].astype('category')

df['Day of Month'] = df.apply(lambda x: x['Date'].day, axis=1)
df['Day of Month'] = df['Day of Month'].astype('category')

df = df.drop(['Date'], axis=1)

df.to_csv('processed_data.csv', index=False)
