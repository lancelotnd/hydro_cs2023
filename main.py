import pandas as pd
import matplotlib as plt
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def main():
    df = pd.read_csv(CSV)

    df = to_float(df)
    print(df)
    #visualize(df)
    df.to_csv("parquet.csv")


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









if __name__ == "__main__":
    main()