import numpy as np 
import pandas as pd
import streamlit as st
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import random
import smtplib as s

def sendmail():
    obj=s.SMTP('smtp.gmail.com',587)
    obj.starttls()
    subjectval="Alert Message"
    bodyval="Fraud Transaction Detected!!. Please Enquire Further"
    message="subject:{}\n\n{}".format(subjectval,bodyval)
    obj.login("faizanali6509@gmail.com","gtuodvlivsfmptdz")
    obj.sendmail("faizanali6509@gmail.com","mdnaveeda02@gmail.com",message)
    obj.quit()


def encode(lst):
    num1 = random.random()
    num2 = random.random()
    num3 = random.random()

    lst[0]=lst[0]+str(num1)
    lst[1]=lst[1]+str(num2)
    lst[2]=lst[2]+str(num3)
    return lst

data=pd.read_csv('card_transdata.csv')

legit = data[data.fraud == 0.0]
fraud = data[data.fraud == 1.0]

# undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=5000, random_state=2)
non_legit_sample=fraud.sample(n=5000, random_state=2)
df = pd.concat([legit_sample, non_legit_sample], axis=0)

y=df[['fraud']]
x=df.drop('fraud',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.90,random_state=2)


log=LogisticRegression(solver='liblinear' )
model=log.fit (x_train,y_train["fraud"])

log=LogisticRegression(max_iter=150 )
model=log.fit (x_train,y_train["fraud"])

st.title("Credit Card Fraud Detection Model")
st.markdown(body='')
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

st.write("Distance from home, Distance from last transaction, Ratio to median purchase price, Repeat retailer, Used chip,Used pin number, Online order")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(r'3.png')

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# input_df_lst=encode(input_df_lst)
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    input_df_lst=encode(input_df_lst)
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0.0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")
        sendmail()

#plotting data
# Define the labels
LABELS = ["Normal", "Fraud"]

# Count the classes
count_classes = pd.value_counts(data['fraud'], sort=True)

# Create the bar plot
fig, ax = plt.subplots()
count_classes.plot(kind='bar', rot=0, ax=ax)
ax.set_title("Transaction Class Distribution")
ax.set_xticks(range(2))
ax.set_xticklabels(LABELS)
ax.set_xlabel("Class")
ax.set_ylabel("Frequency")

# Display the plot in Streamlit
st.pyplot(fig)

#plotting balanced data
# Count the classes
count_classes1 = pd.value_counts(df['fraud'], sort=True)

# Create the bar plot
fig1, ax1 = plt.subplots()
count_classes1.plot(kind='bar', rot=0, ax=ax1)
ax1.set_title("Transaction Class Distribution After Undersampling")
ax1.set_xticks(range(2))
ax1.set_xticklabels(LABELS)
ax1.set_xlabel("Class")
ax1.set_ylabel("Frequency")

# Display the plot in Streamlit
st.pyplot(fig1)