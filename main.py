import pandas as pd
import matplotlib as plt
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor 
import seaborn as sns



PARQUET = "NORWAY_INFLOW_2018-2022.parquet"
CSV = "parquet.csv"

def visualize(features):
    for col in features.columns:
        if col != 'Unnamed: 0':
            print(col)
            features[col].plot()
            plt.title(col)
            plt.savefig("out/"+col+".png")
            plt.close()


def to_float(dataset):
    new_dataset = pd.DataFrame()

    for col in list(dataset.columns):
        new_dataset[col] = dataset[col].astype(float)

    return new_dataset



def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.savefig("00_hist.png")


def historize(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    return hist



def linear_regression_multiple_inputs(normalizer, train_features):

    linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
    ])

    print(linear_model.predict(train_features[:10]))

    



def normalize(train_features, train_labels):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    print(normalizer.mean.numpy())

    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())

    H1006 = np.array(train_features['H1006'])

    h1006_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    h1006_normalizer.adapt(H1006)

    h1006_model = tf.keras.Sequential([
    h1006_normalizer,
    layers.Dense(units=1)
    ])

    h1006_model.summary()

    print(h1006_model.predict(H1006[:10]))


    h1006_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

    history = h1006_model.fit(
    train_features['H1006'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

    historize(history)
    plot_loss(history)

    return normalizer





def main():
    df = pd.read_csv(CSV)

    df = to_float(df)
    print(df)
    #visualize(df)


    #prepare train test dataset
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    #We remove the `entsoe_inflow` label from the dataset as it is the value 
    #we will try to predict


    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('entsoe_inflow')
    test_labels = test_features.pop('entsoe_inflow')

    print(train_dataset.describe().transpose()[['mean', 'std']])

    normalizer  = normalize(train_features, train_labels)

    linear_regression_multiple_inputs(normalizer, train_features)

    
    







if __name__ == "__main__":
    main()