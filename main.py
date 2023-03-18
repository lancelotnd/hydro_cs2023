import pandas as pd
import matplotlib as plt
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor 


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
    visualize(df)
    df.to_csv("parquet.csv")




if __name__ == "__main__":
    main()