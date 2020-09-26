from aux_lib import stats as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


random_list = list(np.random.randint(1,100,10))
random_list_2 = list(np.random.randint(1,100,10))

def test_contagem():
    assert st.contagem(random_list) == len(random_list)

def test_soma_list():
    assert st.soma_list(random_list) == sum(random_list)

def test_media_list():
    assert st.media_list(random_list) == np.mean(random_list)

def test_sample_variance():
    assert st.sample_variance(random_list).round(2) == np.var(random_list,ddof=1).round(2)

def test_sample_covariance():
    #cov1 = np.cov(random_list, random_list_2, ddof=1)
    #print(cov1)
    #print(cov1[0][1])
    assert st.sample_covariance(random_list, random_list_2).round(2) == np.cov(random_list, random_list_2, ddof=1)[0][1].round(2)

def test_FNC_fit():
    alfa1, beta1 = st.FNC_fit(random_list, random_list_2)
    beta2, alfa2 = np.polyfit(random_list, random_list_2, 1)

    #reg = LinearRegression().fit(random_list, random_list_2)
    
    list1 = np.reshape(random_list, (-1,1))
    list2 = np.reshape(random_list_2, (-1,1))
    reg = LinearRegression().fit(list1, list2)
    
    beta3 = reg.coef_
    alfa3 = reg.intercept_

    print(beta3)
    
    assert [alfa1.round(5), beta1.round(5)] == [alfa2.round(5), beta2.round(5)]
    assert [alfa1.round(5), beta1.round(5)] == [alfa3.round(5), beta3.round(5)]

def test_FNC_predict():
    list1 = np.reshape(random_list, (-1,1))
    list2 = np.reshape(random_list_2, (-1,1))
    reg = LinearRegression().fit(list1, list2)

    target_pred1 = st.FNC_predict(random_list, random_list_2)
    target_pred2 = reg.predict(list1)

    print(target_pred1)
    print(target_pred2)
    print(target_pred2[0])
    print(target_pred1[1]==target_pred2[1][0])
    

    y1 = [round(x, 5) for x in target_pred1]
    #y2 = [round(x, 5) for x in target_pred2]
    y2 = []
    for x in target_pred2:
        y2.append( x[0].round(5) )
    
    print(y1[0])
    print(y2[0])
    print(y1[0]==y2[0])

    #assert target_pred1 == target_pred2

    #v1 = [(x==y) for x,y in zip(target_pred1, target_pred2]
    v1 = [(x==y) for x,y in zip(y1, y2)]
    assert all(v1)


def test_FNC_rmse():
    feature = np.reshape(random_list, (-1,1))
    target = np.reshape(random_list_2, (-1,1))
    reg = LinearRegression().fit(feature, target)

    target_pred = reg.predict(feature)

    rmse1 = np.sqrt( mean_squared_error(target_pred, target) )
    rmse2 = st.FNC_rmse(feature, target)

    assert rmse1.round(5) == rmse2.round(5)