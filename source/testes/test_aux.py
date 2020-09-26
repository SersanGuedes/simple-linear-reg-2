import pandas as pd
from aux_lib.aux1 import own_read_csv

def test_own_read_csv(path_file='data/boston1.csv'):
    dict_values = own_read_csv(path_file)

    df_from_dict = pd.DataFrame.from_dict(dict_values).round(2)
    df_from_pandas = pd.read_csv(path_file).round(2)

    return df_from_dict, df_from_pandas
    assert df_from_dict.equals(df_from_pandas) == True