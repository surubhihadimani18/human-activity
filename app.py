import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

df = pd.read_csv("test.csv")

st.sidebar.title("🏃 HAR Dashboard")

page = st.sidebar.selectbox(
    "Choose Analysis",
    [
        "Overview",
        "Activity Analysis",
        "Feature Analysis"
    ]
)

st.title("🏃 Human Activity Recognition Dashboard")

if page == "Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Subjects", df["subject"].nunique())
    col4.metric("Activities", df["Activity"].nunique())

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Information")

    info = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values"
            ],
            "Value": [
                df.shape[0],
                df.shape[1],
                df.isnull().sum().sum()
            ]
        }
    )

    st.dataframe(info, use_container_width=True)

elif page == "Activity Analysis":

    st.header("Activity Analysis")

    activity_counts = df["Activity"].value_counts().reset_index()

    activity_counts.columns = ["Activity", "Count"]

    fig = px.bar(
        activity_counts,
        x="Activity",
        y="Count",
        text="Count",
        title="Activity Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.pie(
        activity_counts,
        names="Activity",
        values="Count",
        title="Activity Percentage"
    )

    st.plotly_chart(fig2, use_container_width=True)

elif page == "Feature Analysis":

    st.header("Feature Analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    feature_columns = []

    for column in numeric_columns:
        if column != "subject":
            feature_columns.append(column)

    selected_feature = st.selectbox(
        "Select a Sensor Feature",
        feature_columns
    )

    fig = px.box(
        df,
        x="Activity",
        y=selected_feature,
        color="Activity",
        title="Feature Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.caption(
    "Built with Python and Streamlit"
)
