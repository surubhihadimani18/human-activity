import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ============================================================

# PAGE CONFIGURATION

# ============================================================

st.set_page_config(
page_title="Human Activity Recognition Dashboard",
page_icon="🏃",
layout="wide",
initial_sidebar_state="expanded"
)

# ============================================================

# CUSTOM CSS

# ============================================================

st.markdown("""

<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    font-weight: 800;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #4CAF50;
}

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

</style>

""", unsafe_allow_html=True)

# ============================================================

# LOAD DATA

# ============================================================

@st.cache_data
def load_data():

```
try:
    data = pd.read_csv("data/test.csv")
except FileNotFoundError:
    data = pd.read_csv("test.csv")

return data
```

df = load_data()

# ============================================================

# SIDEBAR

# ============================================================

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

st.sidebar.markdown("---")

st.sidebar.info(
"""
**Dataset:** Human Activity Recognition

```
Sensor-based activity classification using
smartphone accelerometer and gyroscope data.
"""
```

)

# ============================================================

# HEADER

# ============================================================

st.title("🏃 Human Activity Recognition Dashboard")

st.markdown(
"""
### Deep Analysis of Human Activities Using Smartphone Sensor Data

```
This interactive dashboard explores activity patterns,
sensor features, dimensionality reduction and machine
learning-based activity classification.
"""
```

)

st.markdown("---")

# ============================================================

# OVERVIEW PAGE

# ============================================================

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

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

with col2:

    st.subheader("Activity Categories")

    activities = df["Activity"].value_counts()

    activity_df = pd.DataFrame({
        "Activity": activities.index,
        "Count": activities.values
    })

    fig = px.bar(
        activity_df,
        x="Count",
        y="Activity",
        orientation="h",
        text="Count",
        title="Number of Records per Activity"
    )

    fig.update_layout(
        height=450,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("📋 Dataset Information")

info_df = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Numeric Features",
        "Missing Values",
        "Duplicate Rows"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        len(df.select_dtypes(include=np.number).columns),
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

# ============================================================

# ACTIVITY ANALYSIS

# ============================================================

elif page == "📊 Activity Analysis":

```
st.header("📊 Human Activity Distribution")

activity_counts = df["Activity"].value_counts().reset_index()

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

    fig.update_layout(
        xaxis_tickangle=-30
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
        hole=0.45,
        title="Activity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("📊 Percentage Distribution")

percentage_df = activity_counts.copy()

percentage_df["Percentage"] = (
    percentage_df["Count"]
    / percentage_df["Count"].sum()
    * 100
).round(2)

st.dataframe(
    percentage_df,
    use_container_width=True,
    hide_index=True
)
```

# ============================================================

# SUBJECT ANALYSIS

# ============================================================

elif page == "👥 Subject Analysis":

```
st.header("👥 Subject-wise Activity Analysis")

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
    title="Number of Activity Records per Subject"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.subheader("Activity Distribution by Subject")

subject_activity = (
    df.groupby(
        ["subject", "Activity"]
    )
    .size()
    .reset_index(name="Count")
)

fig = px.bar(
    subject_activity,
    x="subject",
    y="Count",
    color="Activity",
    barmode="stack",
    title="Activity Composition for Each Subject"
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

st.subheader(
    f"Subject {selected_subject} Analysis"
)

subject_activity_counts = (
    subject_data["Activity"]
    .value_counts()
    .reset_index()
)

subject_activity_counts.columns = [
    "Activity",
    "Count"
]

fig = px.pie(
    subject_activity_counts,
    names="Activity",
    values="Count",
    title=f"Activity Distribution for Subject {selected_subject}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ============================================================

# FEATURE ANALYSIS

# ============================================================

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
    "Select a Sensor Feature",
    feature_columns,
    index=0
)

col1, col2 = st.columns(2)

with col1:

    fig = px.box(
        df,
        x="Activity",
        y=selected_feature,
        color="Activity",
        title=f"{selected_feature} Across Activities"
    )

    fig.update_layout(
        xaxis_tickangle=-30
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
        barmode="overlay",
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

st.subheader("Feature Statistics")

statistics = df[selected_feature].describe()

st.dataframe(
    statistics,
    use_container_width=True
)
```

# ============================================================

# CORRELATION ANALYSIS

# ============================================================

elif page == "🔥 Correlation Analysis":

```
st.header("🔥 Feature Correlation Analysis")

st.info(
    "Because the dataset contains hundreds of features, "
    "the dashboard displays a correlation heatmap for a "
    "selected number of features."
)

numeric_features = [
    col for col in df.select_dtypes(include=np.number).columns
    if col != "subject"
]

n_features = st.slider(
    "Number of Features",
    min_value=10,
    max_value=min(50, len(numeric_features)),
    value=25
)

selected_features = numeric_features[:n_features]

corr_matrix = (
    df[selected_features]
    .corr()
)

fig = px.imshow(
    corr_matrix,
    title="Feature Correlation Heatmap",
    aspect="auto"
)

fig.update_layout(
    height=800
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.subheader("Highly Correlated Feature Pairs")

corr_pairs = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

corr_pairs = (
    corr_pairs.stack()
    .reset_index()
)

corr_pairs.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

corr_pairs["Absolute Correlation"] = (
    corr_pairs["Correlation"].abs()
)

top_corr = (
    corr_pairs
    .sort_values(
        "Absolute Correlation",
        ascending=False
    )
    .head(15)
)

st.dataframe(
    top_corr,
    use_container_width=True,
    hide_index=True
)
```

# ============================================================

# PCA VISUALIZATION

# ============================================================

elif page == "🧠 PCA Visualization":

```
st.header("🧠 PCA Dimensionality Reduction")

st.markdown(
    """
    PCA reduces the high-dimensional sensor dataset into
    two principal components, making activity patterns
    easier to visualize.
    """
)

feature_columns = [
    col for col in df.columns
    if col not in ["Activity", "subject"]
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

explained_variance = (
    pca.explained_variance_ratio_
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "PC1 Explained Variance",
        f"{explained_variance[0] * 100:.2f}%"
    )

with col2:

    st.metric(
        "PC2 Explained Variance",
        f"{explained_variance[1] * 100:.2f}%"
    )

fig = px.scatter(
    pca_df,
    x="PCA 1",
    y="PCA 2",
    color="Activity",
    title="Human Activities in PCA Space",
    opacity=0.7
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ============================================================

# MACHINE LEARNING

# ============================================================

elif page == "🤖 Machine Learning":

```
st.header("🤖 Activity Classification Model")

st.markdown(
    """
    Train a Random Forest classifier to predict
    human activity using sensor-based features.
    """
)

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

st.subheader("Model Configuration")

n_estimators = st.slider(
    "Number of Trees",
    min_value=50,
    max_value=300,
    value=150,
    step=50
)

if st.button("🚀 Train Model"):

    with st.spinner("Training Random Forest Model..."):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
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

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Training Samples",
        X_train.shape[0]
    )

    col3.metric(
        "Testing Samples",
        X_test.shape[0]
    )

    st.markdown("---")

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

    st.markdown("---")

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Top 20 Important Features")

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
        .head(20)
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Most Important Features for Activity Prediction"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
```

# ============================================================

# AUTOMATED INSIGHTS

# ============================================================

elif page == "💡 Insights":

```
st.header("💡 Automated Dataset Insights")

total_records = len(df)

most_common_activity = (
    df["Activity"]
    .value_counts()
    .idxmax()
)

least_common_activity = (
    df["Activity"]
    .value_counts()
    .idxmin()
)

most_common_count = (
    df["Activity"]
    .value_counts()
    .max()
)

least_common_count = (
    df["Activity"]
    .value_counts()
    .min()
)

subject_counts = df["subject"].value_counts()

busiest_subject = subject_counts.idxmax()

busiest_subject_count = subject_counts.max()

st.markdown(
    f"""
    ### 🔍 Key Findings

    **1. Dataset Size**

    The dataset contains **{total_records:,} activity records**
    collected from **{df["subject"].nunique()} subjects**.

    **2. Activity Classes**

    The system recognizes **{df["Activity"].nunique()} different
    human activities**, including walking, sitting, standing and
    laying activities.

    **3. Most Frequent Activity**

    **{most_common_activity}** is the most frequently recorded
    activity with **{most_common_count} records**.

    **4. Least Frequent Activity**

    **{least_common_activity}** has the lowest number of records
    with **{least_common_count} samples**.

    **5. Subject Contribution**

    Subject **{busiest_subject}** contributed the highest number
    of observations with **{busiest_subject_count} records**.

    **6. High-Dimensional Sensor Data**

    The dataset contains hundreds of engineered features derived
    from accelerometer and gyroscope signals. These include time
    domain and frequency domain measurements.

    **7. Machine Learning Potential**

    The presence of clearly labeled activities and a large number
    of numerical sensor features makes this dataset suitable for
    supervised machine learning classification.
    """
)

st.markdown("---")

st.subheader("📊 Activity Summary")

summary = (
    df.groupby("Activity")
    .agg(
        Records=("Activity", "count")
    )
    .reset_index()
    .sort_values(
        "Records",
        ascending=False
    )
)

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

# ============================================================

# FOOTER

# ============================================================

st.markdown("---")

st.markdown(
""" <div style="text-align:center"> <h4>🏃 Human Activity Recognition Analytics Dashboard</h4> <p>Built using Python • Streamlit • Pandas • Plotly • Scikit-learn</p> </div>
""",
unsafe_allow_html=True
)
