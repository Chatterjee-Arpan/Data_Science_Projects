# 1 Good (lower risk)
# 0 Bad (higher risk)

import streamlit as st
import pandas as pd
import joblib   

model = joblib.load('best_xgb_model.pkl')

encoders = {col : joblib.load(f'{col}_label_encoder.pkl') for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account']}

st.title('Credit Risk Assessment App')
st.write('Enter applicant details to predict credit risk. Press "Predict" to check the result.')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
sex = st.selectbox('Sex', ['male','female'])
job = st.number_input('Job', min_value=0, max_value=3, value=1)
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox('Saving Accounts', ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox('Checking Account', ['little', 'moderate', 'rich'])
credit_amount = st.number_input('Credit Amount', min_value=0.0, value=1000.0)
duration = st.number_input('Duration (months)', min_value=1, value=12)

input_data = pd.DataFrame({
    'Age': [age],
    'Sex': [encoders['Sex'].transform([sex])[0]],
    'Job': [job],
    'Housing': [encoders['Housing'].transform([housing])[0]],
    'Saving accounts': [encoders['Saving accounts'].transform([saving_accounts])[0]],
    'Checking account': [encoders['Checking account'].transform([checking_account])[0]],
    'Credit amount': [credit_amount],
    'Duration': [duration]
})

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    
    st.subheader(f'Prediction Result: {int(prediction)}')

    if prediction == 1:
        st.success('The applicant is predicted to be a **GOOD credit risk.**')
    else:
        st.error('The applicant is predicted to be a **BAD credit risk.**')