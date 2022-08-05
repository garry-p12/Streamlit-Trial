#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

# In[15]:


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


# In[16]:


with header:
    st.title('Employee Satisfaction level')
    st.header('Are your employees satisfied with their work?')


# In[75]:


with dataset:
    data = pd.read_csv('employees (1) (1) (1) (2).csv')
    data['last_evaluation'].fillna('0',inplace=True)
    data['department'].fillna('support',inplace=True)
    data['satisfaction'].fillna(0,inplace=True)
    data['tenure'].fillna(0,inplace=True)
    le = LabelEncoder()
    label = le.fit_transform(data['EmployeeName'])
    label = le.fit_transform(data['Agency'])
    label = le.fit_transform(data['fname'])
    label = le.fit_transform(data['lname'])
    data.drop("EmployeeName", axis=1, inplace=True)
    data.drop("Agency", axis=1, inplace=True)
    data.drop("fname", axis=1, inplace=True)
    data.drop("lname", axis=1, inplace=True)
    data["EmployeeName"] = label
    data["Agency"] = label
    data["fname"] = label
    data["lname"] = label
    data['status']=data['status'].map({'Employed':1,'Left':0})
    data['salary']=data['salary'].map({'low':0,'medium':1,'high':2})
    data['department']=data['department'].map({'product':0,'sales':1,'support':2,'temp':3,'IT':4,'admin':5,'engineering':6,'finance':7,'information_technology':8,'management':9,'marketing':10,'procurement':11})
    #tenure = tenure.to_frame(name='tenure')  
    data = data.drop(['Agency'],axis=1)
    data = data.drop(['EmployeeName'],axis=1)
    data = data.drop(['fname'],axis=1)
    data = data.drop(['lname'],axis=1)
    data = data.drop(['last_evaluation'],axis=1)
    data = data.drop(['avg_monthly_hrs'],axis=1)
    ten = pd.DataFrame(data['tenure'].value_counts().head(50))
    n_projects = pd.DataFrame(data['n_projects'].value_counts().head(50))
    department = pd.DataFrame(data['department'].value_counts().head(50))
    status = pd.DataFrame(data['status'].value_counts().head(50))
    st.bar_chart(data = ten)
    st.bar_chart(n_projects)
    st.bar_chart(department)
    st.bar_chart(status)


# In[39]:


type(ten)


# In[28]:


#tenure = tenure.to_frame(name='tenure')


# In[76]:


data.head()


# In[81]:


target = np.array(data.drop(['satisfaction'],1))
features = np.array(data['satisfaction'])


# In[84]:


with model_training:
    sel_col,disp_col = st.columns(2)
#     input_feature1 = sel_col.text_input('What feature to use?','tenure')
#     input_feature2 = sel_col.text_input('What feature to use?','rating')
#     input_feature3 = sel_col.text_input('What feature to use?','n_projects')
#     input_feature4 = sel_col.text_input('What feature to use?','salary')
#     input_feature5 = sel_col.text_input('What feature to use?','status')
#     input_feature6 = sel_col.text_input('What feature to use?','age')
#     input_feature7 = sel_col.text_input('What feature to use?','department')
    
    #st.write('You selected:', options)
    x_train , x_test , y_train , y_test = train_test_split(target,features,test_size=0.25,random_state=42)
    gb = GradientBoostingRegressor
    regr = gb(max_depth=100)
    regr.fit(x_train,y_train)
    y_pred = regr.predict(x_test)
    
    def gb_param_selector(tenure,rating,projects,salary,salary1,status,age,department):
        prediction = regr.predict([[tenure, rating, n_projects, salary,salary1,status,age,department]])
        return prediction

    pickle.dump(regr,open('attrition_model.pkl', 'wb')) 
    model = pickle.load(open('attrition_model.pkl','rb'))     
    # following lines create boxes in which user can enter data required to make prediction 
    tenure = st.number_input('Tenure')
    rating = st.number_input('Rating') 
    n_projects = st.number_input("Number of projects completed") 
    salary = st.number_input("Salary level")
    salary1 = st.number_input('Raw salary')
    status = st.number_input('Status')
    age = st.number_input('Age')
    department = st.number_input('Department')
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = gb_param_selector(tenure, rating, n_projects, salary, salary1,status,age,department) 
        st.success('The employee has a satisfaction level of {}'.format(result))
