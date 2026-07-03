import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Predictive Analytics Dashboard",
                   page_icon="📊",
                   layout="wide")

st.title("📊 Predictive Analytics Using Historical Data")
st.write("Sales Forecasting Dashboard")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload Superstore CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv("uploaded_file")

    st.success("Dataset Loaded Successfully!")

    # Date Conversion
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    # Create Date Features
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Day"] = df["Order Date"].dt.day

    # -----------------------------
    # KPI Cards
    # -----------------------------
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${total_sales:,.2f}")
    c2.metric("Total Profit", f"${total_profit:,.2f}")
    c3.metric("Orders", total_orders)
    c4.metric("Customers", total_customers)

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    # -----------------------------
    # Monthly Sales Trend
    # -----------------------------
    st.subheader("📈 Monthly Sales Trend")

    monthly_sales = df.groupby(
        df["Order Date"].dt.to_period("M")
    )["Sales"].sum().reset_index()

    monthly_sales["Order Date"] = monthly_sales["Order Date"].astype(str)

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Category
    # -----------------------------
    st.subheader("📦 Sales by Category")

    category = df.groupby("Category")["Sales"].sum().reset_index()

    fig = px.bar(
        category,
        x="Category",
        y="Sales",
        color="Category",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Region
    # -----------------------------
    st.subheader("🌍 Sales by Region")

    region = df.groupby("Region")["Sales"].sum().reset_index()

    fig = px.pie(
        region,
        names="Region",
        values="Sales",
        hole=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Sub-Category
    # -----------------------------
    st.subheader("📊 Top 10 Sub-Categories")

    sub = (
        df.groupby("Sub-Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        sub,
        x="Sub-Category",
        y="Sales",
        color="Sales",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Sales by Segment
    # -----------------------------
    st.subheader("👥 Sales by Segment")

    seg = df.groupby("Segment")["Sales"].sum().reset_index()

    fig = px.bar(
        seg,
        x="Segment",
        y="Sales",
        color="Segment",
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)
    # -----------------------------
    # Random Forest Prediction
    # -----------------------------
    st.subheader("🤖 Sales Prediction Using Random Forest")

    # Features
    X = df[["Year", "Month", "Day"]]

    # Target
    y = df["Sales"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # -----------------------------
    # Model Evaluation
    # -----------------------------
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.2f}")

    # -----------------------------
    # Actual vs Predicted
    # -----------------------------
    st.subheader("📈 Actual vs Predicted Sales")

    result = pd.DataFrame({
        "Actual Sales": y_test.values,
        "Predicted Sales": predictions
    })

    fig = px.scatter(
        result,
        x="Actual Sales",
        y="Predicted Sales",
        title="Actual vs Predicted Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Prediction Table
    # -----------------------------
    st.subheader("Prediction Results")

    st.dataframe(result.head(20))

    # -----------------------------
    # Download CSV
    # -----------------------------
    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="sales_predictions.csv",
        mime="text/csv"
    )
    # ==============================
# SIDEBAR FILTERS
# ==============================

st.sidebar.title("Dashboard Filters")