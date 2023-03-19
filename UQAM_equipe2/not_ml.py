import pandas as pd
import scipy
from sklearn.metrics import r2_score
import sys
CSV = "parquet.csv"


def to_float(dataset):
    new_dataset = pd.DataFrame()

    for col in list(dataset.columns):
        new_dataset[col] = dataset[col].astype(float)

    return new_dataset


def sum_dataset(dataset):
    dataset['Sum'] = dataset[list(dataset.columns)].sum(axis=1)
    return dataset


def predict_water(dataset, dict_linear,cols): 
    all_preds = []
    for i in dataset.index:
        predictions = []
        for central in cols:
            mb_list = dict_linear[central]
            m = mb_list[0]
            b = mb_list[1]
            x = dataset[central][i]
            predictions.append(m*x+b)
        
        predicted_avg = sum(predictions)/len(predictions)
        all_preds.append(predicted_avg)
    return all_preds




def linear_regression(dataset, target, dict_corr):
    formulas = {}
    for col in list(dataset.columns):
        X = list(dataset[col])
        Y = list(target)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X, Y)
        formulas[col] = [slope, intercept]
    return formulas
        
    

def test_linear(X, target, mb):
    preds = []
    for x in X:
        preds.append(mb[0]*  x + mb[1])
    score = r2_score(target, preds)
    print("The accuracy of this equation model is {}%".format(round(score, 2) *100))


def deduce_factor(dataset, target):
    factor = pd.DataFrame()
    for col in list(dataset.columns):
        factor[col] = target /dataset[col]
        factor[col] = factor[col].astype(int) 
    
    return factor



def display_correlation(dataset, ref, file):
    dict_corr = {}
    best_name = ""
    best_value = 0
    for cols in list(dataset.columns):
        coor = ref.corr(dataset[cols])*100
        dict_corr[cols] = coor
        if coor > best_value:
            best_value = coor
            best_name = cols

    return dict_corr
    



def main(): 
    TEST_FILE = sys.argv[1]
    df = pd.read_csv(CSV)
    df = to_float(df)

    tmp = df.copy()

    not_features = pd.concat([tmp.pop(x) for x in ['year','week','entsoe_inflow','time_id']], axis=1)
    list_centrals= list(tmp.columns)
    features  = tmp.copy()
    sum_data = sum_dataset(tmp)
    tmp
    not_features["Sum"] = tmp["Sum"]

    df["prod"] = tmp["Sum"]
    dict_coor =  display_correlation(features, df["entsoe_inflow"],"correlation.csv")
    factor = deduce_factor(features,df["entsoe_inflow"] )


    targets = df["entsoe_inflow"]
    dict_linear_equations = linear_regression(features,targets,dict_coor)



    #We find the centrals where the correlation is highest
    best_indicators = [x for x in dict_coor.keys() if dict_coor[x] > 85]

    #We test the prediction accross all the dataset
    predictions = predict_water(features, dict_linear_equations, best_indicators)
    print(predictions)
    score = r2_score(df["entsoe_inflow"], predictions)
    print("The accuracy of our model is {}%".format(round(score, 2) *100))

    print("************************************************************")
    print(" > predicting for ", TEST_FILE)

    dataset_test = pd.read_parquet(TEST_FILE)
    dataset_test.to_csv("tmp",index=False)

    dataset_test = pd.read_csv("tmp")
    dataset_test = to_float(dataset_test)
    test_predictions = predict_water(dataset_test, dict_linear_equations, best_indicators)

    #the predicted values
    print(test_predictions)


    

if __name__ == "__main__":
    main()