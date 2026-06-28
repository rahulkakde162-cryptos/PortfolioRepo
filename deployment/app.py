import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="RahulKakde/PortfolioModel",
    filename="best_model.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Purchase")
st.write(""" Please enter the customer information below to predict whether the customer is likely to purchase the tourism package.
""")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

num_persons = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

preferred_star = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

num_trips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    max_value=50,
    value=2
)

num_children = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=1000000,
    step=500
)

# Categorical inputs
contact_type = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Freelancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Fe Male"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married","Unmarried", "Divorced"]
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)
designation = st.selectbox(
    "Designation",
    [
        "Manager",
        "Senior Manager",
        "Executive",
        "AVP",
        "VP"
    ]
)

duration_pitch = st.number_input(
    "Duration Of Pitch",
    min_value=0,
    max_value=1000,
    value=30
)

followups = st.number_input(
    "Number Of Followups",
    min_value=0,
    max_value=20,
    value=3
)

pitch_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1,2,3,4,5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

# Create input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact_type,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "DurationOfPitch": duration_pitch,
    "NumberOfFollowups": followups,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
}])



if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Product Purchased" if prediction == 1 else "Product Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
