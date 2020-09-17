#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import datetime
import time
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import altair as altair
from fbprophet import Prophet
from skopt import BayesSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[3]:


df = pd.ExcelFile(r'Data Set.xlsx')


# In[4]:


df = df.parse('Multivariate_w_PLC')


# In[5]:


df.tail()


# In[6]:


df['Demand_copy'] = df['Demand']


# ## Missing Vales

# In[15]:


target = ['Demand_copy']
##Features you want the algorithm to run on
predictors = ['Warranty_Install_Base_Count', 'Carepack_Install_Base_Count',
       'Contract_Install_Base_Count', 'PL_rate', 'Commodity_rate', 'PLC_rate','demand_lag_1','demand_lag_2','demand_lag_3']

##any categorical columns can be mentioned here
qual_preds = []


# In[16]:


for lag in range(1,9+1):

    demand_lag_str = 'demand_lag_' + str(lag)
    df[demand_lag_str] = df.groupby(['Part_No'])[target].shift(lag)
    df[demand_lag_str] = df[demand_lag_str].fillna(0)


# In[17]:


# Find % of MV in each variable
p_mv_df=pd.DataFrame(df.isnull().sum()*100/df.shape[0]).rename(columns={0:'p_mv'})


# In[18]:


p_mv_df


# In[19]:


def missing_values(df,predictors):
    for col in predictors:
        nan = np.nan
        imputer = SimpleImputer( missing_values=nan,strategy='mean')
        df[i] = imputer.fit_transform(pd.DataFrame(df[col]))
    return(df)
        


# In[20]:


df = missing_values(df,predictors)


# ## Outlier Detection

# In[21]:


df['Order_Date'] = pd.to_datetime(df['Order_Month'],format = '%Y%m')


# In[22]:


df['Month']=df['Order_Date'].apply(lambda x: x.strftime('%m'))


# In[23]:


df1 = pd.DataFrame(df[['Order_Date','Demand']])
df1.columns = ['ds','y']


# In[24]:


def fit_predict_model(dataframe, interval_width = 0.99, changepoint_range = 0.8):
    m = Prophet(daily_seasonality = False, yearly_seasonality = False, weekly_seasonality = False,
                seasonality_mode = 'multiplicative', 
                interval_width = interval_width,
                changepoint_range = changepoint_range)
    m = m.fit(dataframe)
    
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)
    print('Displaying Prophet plot')
    fig1 = m.plot(forecast)
    return forecast
    
pred = fit_predict_model(df1)


# In[25]:


def detect_anomalies(forecast):
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    #forecast['fact'] = df['y']

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    #anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] =         (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] =         (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
    
    return forecasted

pred = detect_anomalies(pred)


# In[26]:


pred[pred['anomaly']==-1]


# ## Modelling

# In[30]:


def demand(df,test_start_month = 201901, test_end_month = 201924,model = RandomForestRegressor(n_estimators=100), scale_x = preprocessing.StandardScaler(),scale_y = preprocessing.StandardScaler(),offset = 8,demand_lag = 9,tuning= True, scale = True,dummy = True,demand_lag_param = True):
    

    df = df.sort_values(['Part_No','Order_Month'])
    df = df.reset_index()
    df = df.drop('index',axis =1 )

    
    
    if dummy == True:
        df_D = pd.get_dummies(df, columns = qual_preds, drop_first= True) 
    else:
        df_D = df
    df_D = df_D.replace(np.nan,0)
    
    cat = list((set(df_D.columns) - set(df.columns)))
    predictors.extend(cat)
    final_vbls =  list(set(predictors) - set(qual_preds))
    
    
    test_data_period = df[(df['Order_Month'] > test_start_month) & (df['Order_Month'] < test_end_month)].sort_values(['Part_No','Order_Month'])
    
    final = pd.DataFrame()
    pred = pd.DataFrame()
    feature_importances = []
    
    feature_importance_matrix = pd.DataFrame()
    
    if tuning == True:
        model = BayesSearchCV(
                estimator = RandomForestRegressor(
                    n_jobs = 1,
                    criterion='mse',
                ),
                search_spaces = {
                    'min_weight_fraction_leaf': (1e-9, 0.5, 'uniform'),
                    'max_depth': (1, 50),
                    'max_leaf_nodes': (2, 20),
                    'min_impurity_decrease': (0.01, 1.0, 'uniform'),
                    'min_impurity_split': (0.01, 1.0, 'uniform'),
                    'min_impurity_decrease': (0.01, 1.0, 'uniform'),
                    'ccp_alpha': (1e-9, 1.0, 'log-uniform'),
                    'n_estimators': (50,300)
                },    
                cv = KFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=42
                ),
                n_iter=20,
                verbose = 1,
                return_train_score = True
            )
        y_train = df[df['Order_Month']<test_start_month][target]
        X_train = df[df['Order_Month']<test_start_month][final_vbls]
        model.fit(X_train, y_train)
        model = model.best_estimator_
        print(model.get_params)
        

    
    for planning_month in test_data_period['Order_Month'].sort_values().unique():
            df_D['Demand_copy'] =df_D['Demand']   
            print('planning_month:',planning_month,'\n')
            if demand_lag_param == True:

                for lag in range(1,demand_lag+1):

                    demand_lag_str = 'demand_lag_' + str(lag)
                    df_D[demand_lag_str] = df_D.groupby(['Part_No'])[target].shift(lag)
                    df_D[demand_lag_str] = df_D[demand_lag_str].fillna(0)


            for month in range(0,offset):
                target_month = planning_month + month
                
                print('target_month:',target_month)
                if target_month>test_end_month:
                    continue
                
                for lag in range(1,demand_lag+1):

                    demand_lag_str = 'demand_lag_' + str(lag)
                    df_D[demand_lag_str] = df_D.groupby(['Part_No'])[target].shift(lag)
                    df_D[demand_lag_str] = df_D[demand_lag_str].fillna(0)
                
                train_data_org = df_D[(df_D['Order_Month']< target_month)]
                train_data = df_D[(df_D['Order_Month'] < target_month)]
                
                test_data = df_D[(df_D['Order_Month'] == target_month)]
                test_data_org = df_D[(df_D['Order_Month'] == target_month)]
                

                
                if scale == True:
                    X_train = pd.DataFrame(scale_x.fit_transform(train_data[final_vbls]))
                    X_train.columns = final_vbls
                    
                    y_train = pd.DataFrame(scale_y.fit_transform(train_data[target]))
                    X_test = pd.DataFrame(scale_x.transform(test_data[final_vbls]))
                    
                    X_test.columns = final_vbls
         
                    
                    
                    model.fit(X_train, y_train)
                    print(model.get_params)
                    
                    y_pred = model.predict(X_test)
                    y_pred = pd.DataFrame(y_pred)
                    y_pred = scale_y.inverse_transform(y_pred)
                    y_pred = pd.Series(pd.DataFrame(y_pred)[0])
                    y_pred = y_pred.round()
                    y_pred[y_pred < 0] = 0
                
                else:
                    y_train = train_data[target]
                    X_train = train_data[final_vbls]
                    X_test  = test_data[final_vbls]
                    model.fit(X_train, y_train)
                    print(model.get_params)
                    
                    y_pred = model.predict(X_test)
                    y_pred = pd.Series(y_pred)
                    y_pred = y_pred.round()
                    y_pred[y_pred < 0] = 0
                
                

                                
                if X_test.shape[0] == 0:
                    continue

                

                print(X_test.columns)


                
              
                
                feature_importances.append(model.feature_importances_)
                

                df_D.loc[test_data.index, target[0]] = y_pred.values
 

                
                
                

                pred = pred.append(pd.DataFrame({ "Part_No": test_data_org['Part_No'].values, 'Actual': test_data_org['Demand'].values,                                                     "Offset": [month] * test_data_org.shape[0],"planning_month": [planning_month]*test_data_org.shape[0],"target_month": [target_month] * test_data_org.shape[0],'Fcst':y_pred}),ignore_index = True)
                
                


    y_true, y_pred = np.array(pred['Actual']), np.array(pred['Fcst'])
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 1-mape


    feature_importances = [sum(x)/len(feature_importances) for x in zip(*feature_importances)]
    feature_importance_matrix = pd.DataFrame({'Columns':X_train.columns, 'f_imp' : feature_importances})

    return pred,df_D,feature_importance_matrix,mape,accuracy     
    
                   


# In[35]:


final = demand(df,test_start_month = 201901, test_end_month = 201912, offset = 4,demand_lag = 9,tuning = False, scale = False,dummy = True,demand_lag_param = True)
    


prediction,feature_data, feature_importance_matrix,mape,accuracy = final







