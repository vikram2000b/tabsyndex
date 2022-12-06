import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import math
import sklearn.metrics as sk
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
import scipy.stats as ss
from dython.nominal import theils_u, compute_associations, numerical_encoding


def tabsyndex(real_data, fake_data, cat_cols, target_col=-1, target_type='regr'):

  def mape (vector_a, vector_b):
    return abs(vector_a-vector_b)/abs(vector_a+1e-6)
    
  scaler = MinMaxScaler()
  real_data_norm = scaler.fit_transform(real_data)
  real_data_norm = pd.DataFrame(real_data_norm, columns=real_data.columns)
  fake_data_norm = scaler.transform(fake_data)
  fake_data_norm = pd.DataFrame(fake_data_norm, columns=fake_data.columns)
  
  def basic_stats():
    real_mean = np.mean(real_data, axis=0)
    fake_mean = np.mean(fake_data, axis=0)

    real_std = np.std(real_data, axis=0)
    fake_std = np.std(fake_data, axis=0)

    real_median = np.median(real_data, axis = 0)
    fake_median = np.median(fake_data, axis = 0)


    mean_mape = np.clip(mape(real_mean, fake_mean), 0, 1)
    score = np.sum(mean_mape)
    std_mape = np.clip(mape(real_std, fake_std), 0, 1)
    score += np.sum(std_mape)
    median_mape = np.clip(mape(real_median, fake_median), 0, 1)
    score += np.sum(median_mape)
    score /= len(real_mean)+len(real_std) + len(real_median)

    score = 1-score if score<=1.0 else 0.0
    #print('1:', score)
    return score

  def corr():
    real_corr = compute_associations(real_data, nominal_columns=cat_cols, theil_u=True).astype(float)
    fake_corr = compute_associations(fake_data, nominal_columns=cat_cols, theil_u=True).astype(float)

    real_log_corr = np.sign(real_corr)*np.log(abs(real_corr))
    fake_log_corr = np.sign(fake_corr)*np.log(abs(fake_corr))

    score = np.sum(np.clip(mape(real_log_corr, fake_log_corr).to_numpy().flatten(), 0, 1))
    n = len(real_data.columns)
    score /= n**2 - n
    score = 1-score if score<=1.0 else 0.0

    #print('2:', score)
    return score

  def ml_efficacy():
    real = numerical_encoding(real_data_norm, nominal_columns=cat_cols)
    fake = numerical_encoding(fake_data_norm, nominal_columns=cat_cols)

    real_x = real.drop(real.columns[target_col], axis=1)
    real_y = real[real.columns[target_col]]
    fake_x = fake.drop(fake.columns[target_col], axis=1)
    fake_y = fake[fake.columns[target_col]]

    if target_type == 'regr':
        r_estimators = [
                    RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
                    Lasso(random_state=42, max_iter=5000),
                    Ridge(alpha=1.0, random_state=42),
                    ElasticNet(max_iter=5000,random_state=42),
                  ]
        f_estimators = copy.deepcopy(r_estimators)

        for estimator in r_estimators:
          #print(estimator)
          estimator.fit(real_x, real_y)
        for estimator in f_estimators:
          #print(estimator)
          estimator.fit(fake_x, fake_y)

        r_rmse = [sk.mean_squared_error(real_y, estimator.predict(real_x), squared=False) for estimator in r_estimators]
        r_rmse += [sk.mean_squared_error(fake_y, estimator.predict(fake_x), squared=False) for estimator in r_estimators]
        f_rmse = [sk.mean_squared_error(real_y, estimator.predict(real_x), squared=False) for estimator in f_estimators]
        f_rmse += [sk.mean_squared_error(fake_y, estimator.predict(fake_x), squared=False) for estimator in f_estimators]

        score = np.sum(np.clip(mape(np.array(r_rmse), np.array(f_rmse)), 0, 1))

    elif target_type == 'class':
        r_estimators = [
                LogisticRegression(multi_class='auto', max_iter=5000, random_state=42),
                RandomForestClassifier(n_estimators=10, random_state=42),
                DecisionTreeClassifier(random_state=42),
                MLPClassifier([50, 50], solver='adam', activation='relu', learning_rate='adaptive', random_state=42)
                ]
        f_estimators = copy.deepcopy(r_estimators)

        for estimator in r_estimators:
          #print(estimator)
          estimator.fit(real_x, real_y)
        for estimator in f_estimators:
          #print(estimator)
          estimator.fit(fake_x, fake_y)

        r_f1 = [sk.f1_score(real_y, estimator.predict(real_x), average='micro') for estimator in r_estimators]
        r_f1 += [sk.f1_score(fake_y, estimator.predict(fake_x), average='micro') for estimator in r_estimators]
        f_f1 = [sk.f1_score(real_y, estimator.predict(real_x), average='micro') for estimator in f_estimators]
        f_f1 += [sk.f1_score(fake_y, estimator.predict(fake_x), average='micro') for estimator in f_estimators]

        score = np.sum(np.clip(mape(np.array(r_f1), np.array(f_f1)), 0, 1))

    score /= 8
    score = 1 - score if score<=1.0 else 0.0
    #print('3:', score)
    return score

  def pmse():
    data = real_data_norm.append(fake_data_norm, ignore_index=True)
    data['target'] = [0]*len(real_data)+[1]*len(fake_data)
    data = data.sample(frac=1)
    x = data.drop('target', axis=1)
    y = data['target']
    #poly = PolynomialFeatures(degree = 2, include_bias=False)
    #x_poly = poly.fit_transform(x)

    estimator = LogisticRegression(max_iter=5000, random_state=42)
    estimator.fit(x, y)
    p = estimator.predict_proba(x)
    p = p[:, 1]

    k = x.shape[1] + 1 #for intercept
    N = len(p)
    c = len(fake_data)/N
    pmse = sk.mean_squared_error(p, [c]*N)
    pmse0 = ((k-1)*(1-c)**2)*c/N

    ratio = pmse/pmse0
    score = math.pow(1.2,-abs(1-ratio))
    #print('4:', ratio, score)
    return score

  def sup_cov(num_bins=20):
    sup = 0
    scaling_factor = len(real_data)/len(fake_data)

    for col in list(real_data.columns):
      col_sup = 0
      non_zero_cat = 0

      if col in cat_cols:
        real_col_num = real_data[col].value_counts()
        fake_col_num = fake_data[col].value_counts()
        
        for i in real_col_num.index:
          if real_col_num.loc[i] != 0:
            non_zero_cat += 1
            col_sup += min((fake_col_num.loc[i]/real_col_num.loc[i])*scaling_factor,2)
        
        col_sup = col_sup/non_zero_cat
        if(col_sup>1):
          col_sup = 1.0

      else:
        real_col, bins = pd.cut(real_data[col], bins=num_bins, ordered=False, 
                              labels=range(num_bins), retbins=True)
        real_col_num = real_col.value_counts()
        fake_col_num = pd.cut(fake_data[col], bins=bins, ordered=False, 
                              labels=range(num_bins)).value_counts()

        for i in real_col_num.index:
          if real_col_num.loc[i] != 0:
            non_zero_cat += 1
            col_sup += min((fake_col_num.loc[i]/real_col_num.loc[i])*scaling_factor, 2)
        
        col_sup = col_sup/non_zero_cat
        if(col_sup>1):
          col_sup = 1.0
      sup += col_sup

    sup /= len(real_data.columns) #average support coverage
    #print('5:', sup)
    return sup
  
  basic_score = basic_stats()
  corr_score = corr()
  ml_score = ml_efficacy()
  pmse_score = pmse()
  sup_score = sup_cov()
  score = (basic_score + corr_score + ml_score + sup_score+ pmse_score)/5

  return {"score": score, "basic_score": basic_score, "corr_score": corr_score, "ml_score": ml_score, "sup_score": sup_score, "pmse_score": pmse_score}
