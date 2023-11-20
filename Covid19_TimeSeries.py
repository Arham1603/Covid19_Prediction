#%%
# Import packages
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from window_generator import WindowGenerator
from tensorflow.keras import layers, optimizers, losses, callbacks

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
# Load dataset with pandas
df = pd.read_csv('cases_malaysia.csv')

selected_columns = ['date', 'cases_new', 'cases_import', 'cases_recovered', 'cases_active']

# Create a new DataFrame with selected columns
df = df[selected_columns]

date_time = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
# %%
# Basic data inspection
plot_cols = ['cases_new']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
# %%
# Train, validation, test split for tme series data
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
# %%
# Data normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
# %%
# Data inspection after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# %%
# Create a single step window
single_step = WindowGenerator(input_width=30, label_width=30, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])
# %%
# Create LSTM model
import tensorflow as tf
from tensorflow import keras

lstm_model = keras.Sequential()
lstm_model.add(keras.layers.LSTM(64, return_sequences=True))
lstm_model.add(keras.layers.Dropout(0.2))
lstm_model.add(keras.layers.LSTM(32, return_sequences=True))
#lstm_model.add(keras.layers.Dropout(0.2))
lstm_model.add(keras.layers.Dense(1))
# %%
# Function to perform model compile and training

# Create a tensorboard callback project
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log_single", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)

MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
  patience=patience,
  mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[tb, early_stopping])
# %%
# Compile the model and train
history_1 = compile_and_fit(lstm_model, single_step)
keras.utils.plot_model(lstm_model)
# %%
# Evaluate the model
print(lstm_model.evaluate(single_step.val))
print(lstm_model.evaluate(single_step.test))
# %%
# Plot the resultt
single_step.plot(model=lstm_model, plot_col='cases_new')
# %%
#multi_window
multi_predict = WindowGenerator(input_width=30, label_width=30, shift=30, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])
# %%
# Build multi step model
multi_lstm = keras.Sequential()
multi_lstm.add(keras.layers.LSTM(64, return_sequences=True))
lstm_model.add(keras.layers.Dropout(0.2))
multi_lstm.add(keras.layers.LSTM(64, return_sequences=False))
multi_lstm.add(keras.layers.Dense(30*1))
multi_lstm.add(keras.layers.Reshape([30,1]))
# %%
# Create a tensorboard callback project
PATH = os.getcwd()
logpath = os.path.join(PATH, "tensorboard_log_multi", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(logpath)

# Compile and train model for multi step
history_2 = compile_and_fit(multi_lstm, multi_predict)
keras.utils.plot_model(multi_lstm)
# %%
# Evaluate the model
print(multi_lstm.evaluate(multi_predict.val))
print(multi_lstm.evaluate(multi_predict.test))
# %%
# Plot the resultt
multi_predict.plot(model=lstm_model, plot_col='cases_new')
# %%
