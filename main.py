import pandas as pd

PARQUET = "NORWAY_INFLOW_2018-2022.parquet"

def main():
    print("Hello")
    df = pd.read_parquet(PARQUET)
    df.to_csv("parquet.csv")


if __name__ == "__main__":
    main()