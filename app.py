import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pandas as pd 
import pickle 

model=tf.keras.models.load_model("model.keras")


with open("label_encoder_gender.pkl","rb")as f1:
    label_enocder_geo=pickle.load(f1)
with open("onehot_encoder_geography.pkl","rb")as f:
    one_hot_goe=pickle.load(f)
with open("scaler.pkl","rb")as f:
    scaler=pickle.load(f)
    
    
##streamlit

st.title("Predict whether the customer will churn or not")

geography=st.selectbox("Goegraphy",one_hot_goe.categories_[0])
gender=st.selectbox("Gender",label_enocder_geo.classes_)
age=st.slider("Age",18,100)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_sal=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])

#prepare the input

input_data=pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_enocder_geo.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_sal]
    
})
#one hot encoding geography column
geo=one_hot_goe.transform([[geography]])
df=pd.DataFrame(geo,columns=one_hot_goe.get_feature_names_out())

#concat

input_data=pd.concat([input_data.reset_index(drop=True),df],axis=1)

#sclae the data

input_data_scaled=scaler.transform(input_data)

#predict churn

if st.button("Predict"):
    prediction=model.predict(input_data_scaled)
    pred_prob=prediction[0][0]

    if pred_prob>0.5:
        st.write("The Customer is likely to churn")
    else:
        st.write("The Customer is not likely to churn")
