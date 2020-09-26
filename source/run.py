from aux_lib.aux1 import own_read_csv
from aux_lib      import stats as st

ind_feature = 0 # 0: 'RM'
                # 1: 'LSTAT'
                # 2: 'PTRATIO'

path_file = 'data/boston1.csv'

dict_values = own_read_csv(path_file)

v_feature = ['RM','LSTAT','PTRATIO']

feature = dict_values[ v_feature[ ind_feature ] ]
target  = dict_values['MEDV']

st.FNC_predict( feature, target )

st.FNC_rmse( feature, target )
