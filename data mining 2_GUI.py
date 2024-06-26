import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/sarahdarlyna/Downloads/cardio_train.csv', sep=';')

numerical_col = ['age', 'weight', 'height', 'ap_hi', 'ap_lo']

for x in ['age']:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df.loc[df[x] < min, x] = np.nan
    df.loc[df[x] > max, x] = np.nan

df_n = df.dropna(axis=0)

for x in ['weight']:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df_n.loc[df[x] < min, x] = np.nan
    df_n.loc[df[x] > max, x] = np.nan

df_n1 = df_n.dropna(axis=0)

for x in ['height']:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df_n1.loc[df[x] < min, x] = np.nan
    df_n1.loc[df[x] > max, x] = np.nan

df_n2 = df_n1.dropna(axis=0)

for x in ['ap_hi']:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df_n2.loc[df[x] < min, x] = np.nan
    df_n2.loc[df[x] > max, x] = np.nan

df_n3 = df_n2.dropna(axis=0)

for x in ['ap_lo']:
    q75, q25 = np.percentile(df.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    df_n3.loc[df[x] < min, x] = np.nan
    df_n3.loc[df[x] > max, x] = np.nan

df = df_n3.dropna(axis=0)

df['age'] = (df['age'] // 365).astype(int)

df = df.drop(['id'], axis=1)

feature_cols = df.columns[:-1]  # all columns except the last one
target_col = df.columns[-1]  # the last column

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the OneR algorithm
def oneR_algorithm(X_train, y_train):
    # Find the best attribute by trying to split on each of them
    best_attribute = None
    best_error_rate = float('inf')
    attribute_rules = {}

    for attribute in X_train.columns:
        # Count the frequency of each value of the attribute
        value_counts = X_train[attribute].value_counts().to_dict()

        # For each attribute value, find the most frequent class
        rules = {}
        errors = 0
        for value, count in value_counts.items():
            most_frequent_class = y_train[X_train[attribute] == value].mode()[0]
            rules[value] = most_frequent_class
            # Count errors for the current attribute
            errors += count - y_train[X_train[attribute] == value].value_counts()[most_frequent_class]

        # Calculate the error rate for the attribute
        error_rate = errors / len(X_train)

        # Update the best attribute if the current one is better
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_attribute = attribute
            attribute_rules = rules

    return best_attribute, attribute_rules

# A function to make predictions using OneR
def oneR_predict(X, best_attribute, rules):
    predictions = []
    for index, row in X.iterrows():
        attribute_value = row[best_attribute]
        predictions.append(rules.get(attribute_value, y_train.mode()[0]))
    return predictions

# Train the OneR classifier
best_attribute, rules = oneR_algorithm(X_train, y_train)
y_pred = oneR_predict(X_test, best_attribute, rules)


def prediction_cvd():

    gender_option = {
        "Male": 1,
        "Female": 2
    }

    cholestrol_option = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }

    glucose_option = {
        "Normal": 1,
        "Above Normal": 2,
        "Well Above Normal": 3
    }

    smoke_option = {
        "Yes": 1,
        "No": 0
    }

    alcohol_option = {
        "Yes": 1,
        "No": 0
    }

    active_option = {
        "Yes": 1,
        "No": 0
    }

    st.header("Cardio❤️‍🩹ascular Disease Diagnosis")

    user_age = st.number_input("Age", min_value=1, max_value=200, value=1)
    user_gender = st.radio("Gender", list(gender_option.keys()))
    user_height = st.number_input("Height (cm)", min_value=50.0, max_value=260.0, value=50.0)
    user_weight = st.number_input("Weight (kg)", min_value=1.0, max_value=700.0, value=1.0)
    user_ap_hi = st.slider("Systolic Pressure", min_value=70, max_value=300)
    user_ap_lo = st.slider("Diastolic Pressure", min_value=40, max_value=200)
    user_cholestrol = st.selectbox("Cholestrol", list(cholestrol_option.keys()))
    user_glucose = st.selectbox("Glucose", list(glucose_option.keys()))
    user_smoke = st.radio("Smoke", list(smoke_option.keys()))
    user_alcohol = st.radio("Alcohol", list(alcohol_option.keys()))
    user_active = st.radio("Are you physically active", list(active_option.keys()))

    user_submit = st.button("Predict")

    # Prepare user data for prediction
    if user_submit:
        # Create a DataFrame using user input
        user_data = pd.DataFrame({
            "age": [user_age],
            "gender": [user_gender],
            "height": [user_height],
            "weight": [user_weight],
            "ap_hi": [user_ap_hi],
            "ap_lo": [user_ap_lo],
            "cholestrol": [user_cholestrol],
            "glucose": [user_glucose],
            "smoke": [user_smoke],
            "alcohol": [user_alcohol],
            "active": [user_active]
        })

        #  Use the trained OneR classifier to make predictions
        new_predictions = oneR_predict(user_data, best_attribute, rules)

        if new_predictions[0] == 1:
            prediction_text = "** AT RISK 💔 -> 🤑 **"
        else:
            prediction_text = "** NOT AT RISK ❤️‍🩹 **"

        st.subheader("Prediction Result:")
        st.write("Based on the provided information, you are", prediction_text, "of having a cardiovascular.")

def intro():
    st.header("Cardiovascular Disease Diagnosis using OneR algorithm")
    st.subheader("By: Lim Vern Sin (0133235), Sarah Darlyna Bt Mohd Radzi(0134768)")
    video_file = open(
        '/Users/sarahdarlyna/Downloads/Coronary Artery Disease Animation.mp4',
        'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    st.caption('Source: https://www.youtube.com/watch?v=WKrVxKJVh00&ab_channel=MichiganMedicine')

    st.markdown('CVD comes in different types')
    image2 = Image.open('/Users/sarahdarlyna/Downloads/UDMI_Cardiovascular-Disease.png')
    st.image(image2, caption='Source: https://www.udmi.net/cardiovascular-disease-risk/')



page_names_to_funcs = {
    "Introduction": intro,
    "Prediction": prediction_cvd,
}

demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()