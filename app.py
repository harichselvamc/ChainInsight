import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import gdown
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to download the LLaMA-2 model
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        try:
            gdown.download(model_url, model_path, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
    else:
        st.info("Model already exists, skipping download...")

# Function to generate explanations using LLaMA-2
def generate_explanation(concept, data_summary):
    try:
        model_url = 'https://drive.google.com/uc?id=1-4krkrTTx3DvISRrbftG4WhQj6dlAM_E'
        model_path = 'llama-2-7b-chat.ggmlv3.q8_0.bin'

        download_model(model_url, model_path)

        llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={
                "temperature": 0.08,
                "max_new_tokens": 512
            }
        )

        template = """
            Explain the following data summary about {concept} in the context of supply chain management in three lines:
            {data_summary}
            """

        prompt = PromptTemplate(
            input_variables=["concept", "data_summary"],
            template=template
        )

        response = llm.invoke(
            prompt.format(
                concept=concept,
                data_summary=data_summary
            )
        )

        return response.strip()  # Strip leading and trailing whitespace
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app

st.title("Supply Chain Analysis and Forecasting App")

st.write("""
This app performs various supply chain analysis tasks such as data cleaning, 
risk assessment, EOQ calculation, revenue analysis, lead time analysis, demand forecasting, 
and optimal production volume prediction.
""")

# Load Data
data = pd.read_csv("supply_chain_data.csv")

# Data Preprocessing
st.header("Data Preprocessing")
st.write("Initial Data:")
st.write(data.head())

# Handling missing values and duplicates
missing_values = data.isnull().any(axis=1)
duplicate_values = data[data.duplicated()]

st.write("Rows with missing values:")
st.write(data[missing_values])

st.write("Duplicate Rows:")
st.write(duplicate_values)

data.dropna(axis=0, inplace=True)
data.drop_duplicates(inplace=True)

# Risk Assessment
st.header("Risk Assessment")
selected_columns = ["SKU", "Lead times", "Stock levels"]
risk_data = data[selected_columns]
risk_data["Risk Score"] = risk_data["Lead times"] * (1 - risk_data["Stock levels"])
risk_data = risk_data.sort_values(by="Risk Score", ascending=False)

st.write("Top 15 high risk SKUs")
st.write(risk_data.head(15))

risk_data_summary = risk_data.head(15).to_string()
risk_explanation = generate_explanation("risk assessment", risk_data_summary)
st.write("AI Explanation:")
st.write(risk_explanation)

# EOQ Calculation
st.header("Economic Order Quantity (EOQ) Calculation")
holdingcost = 0.2

def calculate_eoq(data):
    D = data["Number of products sold"]
    S = data["Costs"]
    H = data["Number of products sold"] * holdingcost
    EOQ = np.sqrt((2 * S * D) / H)
    return EOQ

data["EOQ"] = calculate_eoq(data)
data["Current Order Quantity"] = data["Order quantities"]

comparison_columns = ["SKU", "EOQ", "Current Order Quantity"]
st.write(
    """### Economic Order Quantity (EOQ) Formula

The EOQ formula is used to calculate the optimal order quantity that minimizes the total inventory costs, which include ordering costs and holding costs. The formula is given by:

EOQ = √(2DS / H)

where:
- \( D \) = Annual demand (units per year)
- \( S \) = Ordering cost per order (cost per order)
- \( H \) = Holding cost per unit per year (cost per unit per year)

### Explanation:
- **Demand (D)**: The total quantity of units required per year.
- **Ordering Cost (S)**: The cost associated with placing a single order, which may include administrative costs, shipping, and handling.
- **Holding Cost (H)**: The cost to keep one unit of inventory in storage for a year, including warehousing costs, insurance, and opportunity costs.

### Example:
If a company has an annual demand (D) of 1000 units, an ordering cost (S) of $50 per order, and a holding cost (H) of $2 per unit per year, the EOQ is calculated as follows:

EOQ = √(2 × 1000 × 50 / 2) = √(50000) ≈ 223.61

This means the optimal order quantity is approximately 224 units per order to minimize total inventory costs.

    """
)
st.write("EOQ vs Current Order Quantity")
st.write(data[comparison_columns])

eoq_data_summary = data[comparison_columns].head(15).to_string()
eoq_explanation = generate_explanation("economic order quantity (EOQ)", eoq_data_summary)
st.write("AI Explanation:")
st.write(eoq_explanation)

# Revenue Analysis
st.header("Revenue Analysis")
mean_revenue = data.groupby(['Customer demographics', 'Product type'])['Revenue generated'].mean().reset_index()
sum_revenue = data.groupby(['Customer demographics', 'Product type'])['Revenue generated'].sum().reset_index()

st.write("Mean Revenue For Each Customer Demographics")
st.write(mean_revenue)
st.write("Sum Revenue For Each Customer Demographics")
st.write(sum_revenue)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(mean_revenue["Customer demographics"] + '-' + mean_revenue["Product type"], mean_revenue["Revenue generated"])
ax.set_xlabel("Customer Demographics - Product Type")
ax.set_ylabel("Mean Revenue")
ax.set_title("Mean Revenue by Customer Demographics & Product Type")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sum_revenue["Customer demographics"] + '-' + sum_revenue["Product type"], sum_revenue["Revenue generated"])
ax.set_xlabel("Customer Demographics - Product Type")
ax.set_ylabel("Sum Revenue")
ax.set_title("Sum Revenue by Customer Demographics & Product Type")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

revenue_data_summary = sum_revenue.to_string()
revenue_explanation = generate_explanation("revenue analysis", revenue_data_summary)
st.write("AI Explanation:")
st.write(revenue_explanation)

# Lead Time Analysis
st.header("Lead Time Analysis")
lead_times_column = "Lead times"
transportation_modes_column = "Transportation modes"
routes_column = "Routes"

average_lead_time_by_mode = data.groupby(transportation_modes_column)[lead_times_column].mean().reset_index()
best_transportation_mode = average_lead_time_by_mode.loc[average_lead_time_by_mode[lead_times_column].idxmin()]
best_mode = data[data[transportation_modes_column] == best_transportation_mode[transportation_modes_column]]
average_lead_time_by_route = data.groupby(routes_column)[lead_times_column].mean().reset_index()
best_route = average_lead_time_by_route.loc[average_lead_time_by_route[lead_times_column].idxmin()]

st.write("Average Lead Times by Transportation Mode:")
st.write(average_lead_time_by_mode)
st.write("The Best Transportation Mode (Shortest Average Lead Time):")
st.write(best_transportation_mode)
st.write("The Average Lead Times by Route within the Best Transportation Mode:")
st.write(average_lead_time_by_route)
st.write("The Best Routes (Shortest Average Lead Times) within the Best Transportation Mode:")
st.write(best_route)

lead_time_data_summary = average_lead_time_by_mode.to_string() + "\n" + average_lead_time_by_route.to_string()
lead_time_explanation = generate_explanation("lead time analysis", lead_time_data_summary)
st.write("AI Explanation:")
st.write(lead_time_explanation)

# Demand Forecasting using LightGBM and AI
st.header("Demand Forecasting using LightGBM and AI")
target_column = "Number of products sold"
features = ['Price', 'Availability', 'Stock levels', 'Lead times', 'Order quantities']

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_column], test_size=0.2, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

num_round = 100
bst = lgb.train(params, train_data, num_round)
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

st.write("Forecasted Customer Demand:")
st.write(y_pred)

forecast_data_summary = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head(15).to_string()
forecast_explanation = generate_explanation("demand forecasting", forecast_data_summary)
st.write("AI Explanation:")
st.write(forecast_explanation)

# Optimal Production Volume Prediction using TensorFlow and AI
st.header("Optimal Production Volume Prediction using TensorFlow and AI")
target_column = "Manufacturing costs"
feature_column = "Production volumes"

X = data[feature_column].values.reshape(-1, 1)
y = data[target_column].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, verbose=0)

min_production_volume = int(data["Order quantities"].min())
max_production_volume = 1000
step_size = 10

cheapest_cost = float("inf")
best_production_volume = None

for production_volume in range(min_production_volume, max_production_volume + 1, step_size):
    normalized_production_volume = scaler.transform(np.array([[production_volume]]))
    predicted_cost = model.predict(normalized_production_volume)
    if predicted_cost[0][0] < cheapest_cost:
        cheapest_cost = predicted_cost[0][0]
        best_production_volume = production_volume

st.write("Most Optimal Production Volume to Minimize Manufacturing Cost:")
st.write(best_production_volume)
st.write("The Cheapest Manufacturing Cost:")
st.write(cheapest_cost)

production_data_summary = f"Optimal Production Volume: {best_production_volume}\nCheapest Manufacturing Cost: {cheapest_cost}"
production_explanation = generate_explanation("optimal production volume prediction", production_data_summary)
st.write("AI Explanation:")
st.write(production_explanation)

# Cross-validation for LightGBM
st.header("Cross-validation for LightGBM")
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
mse_scores = []

for train_index, test_index in kf.split(data):
    train_data = data.loc[train_index, features]
    train_target = data.loc[train_index, target_column]
    test_data = data.loc[test_index, features]
    test_target = data.loc[test_index, target_column]

    train_data_lgb = lgb.Dataset(train_data, label=train_target)
    bst = lgb.train(params, train_data_lgb, num_round)
    y_pred = bst.predict(test_data, num_iteration=bst.best_iteration)
    mse = mean_squared_error(test_target, y_pred)
    mse_scores.append(mse)

average_mse = sum(mse_scores) / num_folds
st.write("Average MSE:")
st.write(average_mse)
cross_validation_summary = f"Cross-validation MSE Scores: {mse_scores}\nAverage MSE: {average_mse}"
cross_validation_explanation = generate_explanation("cross-validation for LightGBM", cross_validation_summary)
st.write("AI Explanation:")
st.write(cross_validation_explanation)
