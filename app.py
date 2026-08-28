import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
ConfusionMatrixDisplay
)

# ------------------------------------------------------------

# PAGE CONFIGURATION

# ------------------------------------------------------------

st.set_page_config(
page_title="Human Activity Recognition Dashboard",
page_icon="🏃",
layout="wide"
)

# ------------------------------------------------------------

# LOAD DATA

# ------------------------------------------------------------

@st.cache_data
def load_data():
return pd.read_csv("test.csv")

df = load_data()

# ------------------------------------------------------------

# SIDEBAR

# ------------------------------------------------------------

st.sidebar.title("🏃 HAR Dashboard")

page = st.sidebar.radio(
"Navigation",
[
"🏠 Overview",
"📊 Activity Analysis",
"👥 Subject Analysis",
"📈 Feature Analysis",
"🔥 Correlation Analysis",
"🧠 PCA Visualization",
"🤖 Machine Learning",
"💡 Insights"
]
)

# ------------------------------------------------------------

# HEADER

# ------------------------------------------------------------

st.title("🏃 Human Activity Recognition Dashboard")

st.write(
"Interactive analysis and machine learning dashboard "
"for Human Activity Recognition using sensor data."
)

# ------------------------------------------------------------

# OVERVIEW

# ------------------------------------------------------------

if page == "🏠 Overview":

```
st.header("📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{df.shape[0]:,}"
)

col2.metric(
    "Total Features",
    df.shape[1] - 2
)

col3.metric(
    "Subjects",
    df["subject"].nunique()
)

col4.metric(
    "Activities",
    df["Activity"].nunique()
)

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

st.subheader("Dataset Information")

info_df = pd.DataFrame({
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

st.dataframe(
    info_df,
    use_container_width=True,
    hide_index=True
)
```

# ------------------------------------------------------------

# ACTIVITY ANALYSIS

# ------------------------------------------------------------

elif page == "📊 Activity Analysis":

```
st.header("📊 Activity Analysis")

activity_counts = (
    df["Activity"]
    .value_counts()
    .reset_index()
)

activity_counts.columns = [
    "Activity",
    "Count"
]

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        activity_counts,
        x="Activity",
        y="Count",
        text="Count",
        title="Activity Frequency Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.pie(
        activity_counts,
        names="Activity",
        values="Count",
        hole=0.4,
        title="Activity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.subheader("Activity Summary")

activity_counts["Percentage"] = (
    activity_counts["Count"]
    / activity_counts["Count"].sum()
    * 100
).round(2)

st.dataframe(
    activity_counts,
    use_container_width=True,
    hide_index=True
)
```

# ------------------------------------------------------------

# SUBJECT ANALYSIS

# ------------------------------------------------------------

elif page == "👥 Subject Analysis":

```
st.header("👥 Subject-wise Analysis")

subject_counts = (
    df.groupby("subject")
    .size()
    .reset_index(name="Records")
)

fig = px.bar(
    subject_counts,
    x="subject",
    y="Records",
    text="Records",
    title="Records per Subject"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

selected_subject = st.selectbox(
    "Select a Subject",
    sorted(df["subject"].unique())
)

subject_data = df[
    df["subject"] == selected_subject
]

subject_activity = (
    subject_data["Activity"]
    .value_counts()
    .reset_index()
)

subject_activity.columns = [
    "Activity",
    "Count"
]

fig = px.pie(
    subject_activity,
    names="Activity",
    values="Count",
    title=f"Activity Distribution for Subject {selected_subject}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ------------------------------------------------------------

# FEATURE ANALYSIS

# ------------------------------------------------------------

elif page == "📈 Feature Analysis":

```
st.header("📈 Sensor Feature Analysis")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

feature_columns = [
    col for col in numeric_columns
    if col != "subject"
]

selected_feature = st.selectbox(
    "Select a Feature",
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

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        df,
        x=selected_feature,
        color="Activity",
        nbins=40,
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.subheader("Feature Statistics")

st.dataframe(
    df[selected_feature].describe()
)
```

# ------------------------------------------------------------

# CORRELATION ANALYSIS

# ------------------------------------------------------------

elif page == "🔥 Correlation Analysis":

```
st.header("🔥 Correlation Analysis")

numeric_features = [
    col for col in df.select_dtypes(
        include=np.number
    ).columns
    if col != "subject"
]

number_of_features = st.slider(
    "Number of Features",
    min_value=10,
    max_value=min(40, len(numeric_features)),
    value=20
)

selected_features = (
    numeric_features[:number_of_features]
)

correlation_matrix = (
    df[selected_features]
    .corr()
)

fig = px.imshow(
    correlation_matrix,
    title="Feature Correlation Heatmap",
    aspect="auto"
)

fig.update_layout(
    height=700
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ------------------------------------------------------------

# PCA VISUALIZATION

# ------------------------------------------------------------

elif page == "🧠 PCA Visualization":

```
st.header("🧠 PCA Visualization")

feature_columns = [
    col for col in df.columns
    if col not in ["Activity", "subject"]
]

X = df[feature_columns]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(
    n_components=2
)

pca_result = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame({
    "PCA 1": pca_result[:, 0],
    "PCA 2": pca_result[:, 1],
    "Activity": df["Activity"]
})

col1, col2 = st.columns(2)

col1.metric(
    "PC1 Variance",
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

col2.metric(
    "PC2 Variance",
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

fig = px.scatter(
    pca_df,
    x="PCA 1",
    y="PCA 2",
    color="Activity",
    title="Activity Clusters using PCA",
    opacity=0.7
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ------------------------------------------------------------

# MACHINE LEARNING

# ------------------------------------------------------------

elif page == "🤖 Machine Learning":

```
st.header("🤖 Machine Learning Model")

feature_columns = [
    col for col in df.columns
    if col not in ["Activity", "subject"]
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

number_of_trees = st.slider(
    "Number of Trees",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)

if st.button("🚀 Train Model"):

    with st.spinner("Training model..."):

        model = RandomForestClassifier(
            n_estimators=number_of_trees,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

    st.success("Model Training Completed!")

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

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot(
        ax=ax,
        xticks_rotation=45
    )

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.subheader("Top 15 Important Features")

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Important Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
```

# ------------------------------------------------------------

# INSIGHTS

# ------------------------------------------------------------

elif page == "💡 Insights":

```
st.header("💡 Automated Insights")

activity_counts = df["Activity"].value_counts()

most_common = activity_counts.idxmax()
least_common = activity_counts.idxmin()

busiest_subject = (
    df["subject"]
    .value_counts()
    .idxmax()
)

st.success(
    f"The dataset contains {df.shape[0]:,} records "
    f"and {df['Activity'].nunique()} activity categories."
)

st.info(
    f"The most common activity is "
    f"'{most_common}' with "
    f"{activity_counts.max()} records."
)

st.warning(
    f"The least common activity is "
    f"'{least_common}' with "
    f"{activity_counts.min()} records."
)

st.write(
    f"👤 **Subject {busiest_subject}** contributed "
    f"the highest number of records."
)

st.subheader("Activity Distribution Summary")

summary = (
    df["Activity"]
    .value_counts()
    .reset_index()
)

summary.columns = [
    "Activity",
    "Records"
]

summary["Percentage"] = (
    summary["Records"]
    / summary["Records"].sum()
    * 100
).round(2)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)
```

# ------------------------------------------------------------

# FOOTER

# ------------------------------------------------------------

st.markdown("---")

st.markdown(
""" <div style='text-align: center;'> <h4>🏃 Human Activity Recognition Dashboard</h4> <p>Built with Python, Streamlit, Plotly and Scikit-learn</p> </div>
""",
unsafe_allow_html=True
)
