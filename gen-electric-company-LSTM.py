import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/home/jothammasila/Projects/Datasets/GE.csv')

# Separate Date for future plotting
train_dates = pd.to_datetime(df['Date'])

# Training variables
cols = list(df)[1:6]

df_for_train = df[cols].astype(float)

# LSTM uses Sigmoid and tanh that are sensitive to magnitude so values need to be 
# normalizes before traing. We use the fit-transform form StandardScaler class.
scaler = StandardScaler()

scaler = scaler.fit(df_for_train)

df_for_train_scaled = scaler.transform(df_for_train)

# As required by the LSTM networks, we require to reshape on input data into
# n_samples x timesteps
# In this example, the n_features is 2, we will make the timesteps =3

train_X = []
train_Y = []

n_future =1 # Number od days to predict into the future.
n_past = 14 # Number of past days used to predict the future

for i in range(n_past, len(df_for_train_scaled -n_future+1)):
    train_X.append(df_for_train_scaled[i-n_past:i, 0:df_for_train.shape[1]])
    train_Y.append(df_for_train_scaled[i+n_future-1:i+n_future, 0])
    
train_X, train_Y = np.array(train_X), np.array(train_Y)

print(f"Train X shape == {train_X.shape}")
print(f"Train Y shape == {train_Y.shape}")


model = Sequential()
model.add(LSTM(64,activation ='relu', input_shape = (train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(train_Y.shape[1]))

model.compile(optimizer='adam',loss='mse')
model.summary()

# Fit model
model.fit(train_X,train_Y,epochs=10, batch_size=8, validation_split=0.1,verbose=1)

# Forecasting
n_future = 90
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(train_X[-n_future])

forecast_copies =np.repeat(forecast, df_for_train.shape[1],axis=1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]


forecast_dates =[]

for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({
    'Date': np.array(forecast_dates),
    'Open': y_pred_future
})

df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

original = df[['Date','Open']]
original['Date'] = pd.to_datetime(original['Date'])
original = original.loc[original['Date']>='2023-5-1']

sns.lineplot(original['Date'],original['Open'])
sns.lineplot(df_forecast['Date'],df_forecast['Open'])




    










