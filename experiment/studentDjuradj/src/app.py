import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import plotly.express as px
from streamlit_cookies_manager import EncryptedCookieManager
import pymongo
import json
init = st.title("Initalizing...")
# This should be on top of your script
cookies = EncryptedCookieManager(
    # This prefix will get added to all your cookie names.
    # This way you can run your app on Streamlit Cloud without cookie name clashes with other apps.
    prefix="localhost/",
    #prefix="",   # no prefix will show all your cookies for this domain
    # You should setup COOKIES_PASSWORD secret if you're running on Streamlit Cloud.
    #password=os.environ.get("COOKIES_PASSWORD", "My secret password"),
    password='123456',
)
# Avoid cookies doesn't work
if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.stop() # shutdown current app
# Read the mongodb configurations
config = json.loads(open("../config/mongo.json").read())
# load sklearn model we've already trained
model = joblib.load("../model/model.pkl")
# setup a mongo client which connect to the database server we 've sepecified
client = pymongo.MongoClient(config['ip'],config['port'])
# switching to db and collection 
db = client[config['db']]
collection = db[config['collection']]
# fetch data from collection with find_one() function intergrted into collection object
cache_data = collection.find_one()
# since find_one() function doesn't like native agregate pipeline in mongodb
# the "_id" columns is as returned,so we need to delete from cache due to useless
del cache_data["_id"]
init.title("")
# beuild a pandas dataframe object with cached data
cache_df = pd.DataFrame(cache_data)
# prebuild the columns which cotained the last day missed students volume
last_day_missed = []
for i in range(len(cache_df)):
    if i == 0:
        last_day_missed.append(0)
    else:
        last_day_missed.append(cache_df["Total sessions missed"][i-1])
cache_df["Last day missed"] = last_day_missed
# initailize a streamlit sidebar component
with st.sidebar:
    # use option_menu() function to initalize an option_menu component
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Fetchdata","Prediction", "Report"],  # required
        icons=["table","gear", "book"],  # optional
        default_index=0,  # optional
    )
if selected == "Fetchdata": # in this page to fetch data from whatever database or from input directly
    choice = st.selectbox(
        "Which mode do you want use?",
        ("by input","by db")
    )
    if choice == "by input":# when selectbox value equals by input
        st.title("Please set the attributes below")
        id = st.text_input("course id")
        last_day_missed = st.number_input("Last day missed",key="last_day_missed",format="%i")
        expecting = st.number_input("Expecting attend number",key="expecting",format="%i")
        button = st.button("Predict")
        progress = st.progress(0)
        if button:
            # predict with the pycaret pipeline
            result = model.predict([[last_day_missed,expecting]])           
            progress.progress(100)
            if result[0][-1] > int(expecting) or result[0][0] < 0:
                st.error("Expectation is too low, please check again")
            else:
                # set cookies with key and value
                cookies['predAttend'] = str(result[0][-1])
                cookies['predMissed'] = str(result[0][0])
                cookies['expecting'] = str(expecting)
                cookies['LastDayMissed'] = str(last_day_missed)
                cookies.save()
                st.success("Result generated,go predictions to check out.")
    if choice == "by db": # when selectbox value equals by db
        st.title("Please set the attributes below")
        course_id = st.text_input("course id")    
        progress = st.progress(0)
        query  = st.button("query it!")
        if query:
          inready_data = cache_df[cache_df["LSOA11"] == course_id]
          last_day_missed = int(inready_data['Last day missed'].values.tolist()[-1])
          expecting = int(inready_data['Total of possible sessions'].values.tolist()[-1])
          result = model.predict([[last_day_missed,expecting]])
          progress.progress(100)
          if result[-1][-1] > inready_data['Total of possible sessions'].values[-1] or result[-1][0] < 0:
              st.error("Expectation is too low,please check again")
          else:
              cookies['predAttend'] = str(result[-1][-1])
              cookies['predMissed'] = str(result[-1][0])
              cookies['expecting'] = str(inready_data['Total of possible sessions'].values[-1])
              cookies['LastDayMissed'] = str(inready_data['Total of possible sessions'].values[0])
              cookies.save()
              st.success("Result generated,go predictions to check out.")
if selected == "Prediction":# In this page is all about visualize predictions
    st.title("Predictions")
    if cookies is None: # cookies checking
        st.warning("Please set the attributes first")
    else:
        st.success("For more details, please check the report section")
        st.plotly_chart( # plot the predictions with plotly express
            px.bar( # initailize a bar chart
                x=["Total of possible sessions", "predAttend", "predMissed"],
                y=[int(float(cookies['expecting'])),int(float(cookies['predAttend'])),int(float(cookies['predMissed']))],
                title="Prediction",
                labels={"x": "Attributes", "y": "Number of attendances"},
            )
        )
if selected == "Report": # In this page to generate a report which describe predicitons
    st.title("Auto generated Report")
    attendance = int(float(cookies['predAttend'])) / (int(float(cookies['predAttend']))+int(float(cookies['predMissed']))) # calculate attendance ratio
    attendance = round(attendance,2)
    miss = round(1 - attendance,2) # calculate miss ratio
    st.write("Hi teacher,this the attendence report for your class")
    st.write("According to your input,model has predict that attendance rate is "+str(attendance))
    col1,col2 = st.columns(2) # split out two columns to show this two metris
    with col2:
      st.metric("Missed",str(miss * 100)+"%")
    with col1:
      st.metric("Attendance",str(attendance * 100)+"%")
    if attendance > 0.8:
        st.write("The attendance rate is higher than 80%, which is good,keep it and let more student enjoy learning.👍")
    else:
        st.write("The attendance rate is lower than 80%, which is not good, please check the reason and try to improve it.😒")
    
