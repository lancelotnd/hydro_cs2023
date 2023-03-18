import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Charger l'ensemble de données
data = pd.read_parquet('NORWAY_INFLOW_2018-2022.parquet')

# Prétraiter les données
data['Time_id'] = data['Year'].astype(str) + '-' + data['Week'].astype(str)
data['Time_id'] = pd.to_datetime(data['Time_id'] + '-1', format='%Y-%W-%w')


def main():
    print("Hello")


if __name__ == "__main__":
    main()