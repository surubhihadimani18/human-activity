import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(
page_title="Human Activity Recognition",
page_icon="🏃",
layout="wide"
)

@st.cache_data
def load_data():
     return pd.read_csv("test.csv")

df = load_data()

st.sidebar.title("🏃 HAR Dashboard")

page = st.sidebar.selectbox(
"Choose Analysis",
[
"Overview",
"Activity Analysis",
"Feature Analysis",
"PCA Analysis",
"Machine Learning"
]
)

st.title("🏃 Human Activity Recognition Dashboard")

st.write(
"Interactive dashboard for analyzing human activities using sensor data."
)

if page == "Overview":

```
st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Subjects", df["subject"].nunique())
col4.metric("Activities", df["Activity"].nunique())

st.subheader("Dataset Preview")

st.dataframe(df.head(20), use_container_width=True)

st.subheader("Dataset Information")

info = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Rows"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum()
    ]
})

st.dataframe(info, use_container_width=True)
```

elif page == "Activity Analysis":

```
st.header("📈 Activity Analysis")

activity_counts = df["Activity"].value_counts().reset_index()

activity_counts.columns = ["Activity", "Count"]

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        activity_counts,
        x="Activity",
        y="Count",
        text="Count",
        title="Activity Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.pie(
        activity_counts,
        names="Activity",
        values="Count",
        title="Activity Percentage"
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Activity Summary")

st.dataframe(activity_counts, use_container_width=True)
```

elif page == "Feature Analysis":

```
st.header("🔍 Feature Analysis")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

feature_columns = [
    column
    for column in numeric_columns
    if column != "subject"
]

selected_feature = st.selectbox(
    "Select a Sensor Feature",
    feature_columns
)

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        df,
        x="Activity",
        y=selected_feature,
        color="Activity",
        title=f"{selected_feature} by Activity"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:

    fig = px.histogram(
        df,
        x=selected_feature,
        color="Activity",
        nbins=30,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(fig, use_container_width=True)
```

elif page == "PCA Analysis":

```
st.header("🧠 PCA Analysis")

feature_columns = [
    column
    for column in df.columns
    if column not in ["Activity", "subject"]
]

X = df[feature_columns]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PCA 1": pca_result[:, 0],
    "PCA 2": pca_result[:, 1],
    "Activity": df["Activity"]
})

st.metric(
    "Total Explained Variance",
    f"{sum(pca.explained_variance_ratio_) * 100:.2f}%"
)

fig = px.scatter(
    pca_df,
    x="PCA 1",
    y="PCA 2",
    color="Activity",
    title="Activity Visualization using PCA"
)

st.plotly_chart(fig, use_container_width=True)
```

elif page == "Machine Learning":

```
st.header("🤖 Machine Learning Model")

feature_columns = [
    column
    for column in df.columns
    if column not in ["Activity", "subject"]
]

X = df[feature_columns]
y = df["Activity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

if st.button("🚀 Train Random Forest Model"):

    with st.spinner("Training model..."):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

    st.success("Model training completed!")

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    st.pyplot(fig)
```

st.markdown("---")

st.markdown(
"### 🏃 Human Activity Recognition Dashboard"
)

st.caption(
"Built using Python, Streamlit, Plotly and Scikit-learn"
)
