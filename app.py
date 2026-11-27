
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ShopSense AI | Purchase Prediction",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d18;
    --panel: #101827;
    --border: #26354b;
    --text: #f1f5f9;
    --muted: #91a0b7;
    --accent: #5b8cff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8,13,24,0.9);
}

[data-testid="stSidebar"] {
    background: #0d1523;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div {
    padding-top: 25px;
}

[data-testid="stSidebar"] * {
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: white;
}

.brand span {
    color: #6ea2ff;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background:
      radial-gradient(circle at 85% 15%, rgba(70,123,255,.22), transparent 35%),
      linear-gradient(120deg, #14223a, #101b2d 65%, #14233d);
    border: 1px solid #2a4164;
    margin-bottom: 25px;
}

.eyebrow {
    color: #80aaff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 32px;
    line-height: 1.3;
    font-weight: 800;
    color: white;
    margin: 12px 0;
}

.hero p {
    color: #aabbd2;
    font-size: 14px;
    line-height: 1.7;
}

.kpi {
    background: linear-gradient(145deg, #152238, #101827);
    border: 1px solid #293c58;
    border-radius: 14px;
    padding: 18px;
    min-height: 115px;
}

.kpi-label {
    color: #9aabc3;
    font-size: 12px;
}

.kpi-value {
    color: white;
    font-size: 26px;
    font-weight: 800;
    margin-top: 12px;
}

.info-panel {
    background: #111d2e;
    border: 1px solid #293b54;
    border-radius: 14px;
    padding: 20px;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f5f7fb;
    margin: 20px 0 12px;
}

.section-caption {
    color: #91a0b7;
    font-size: 13px;
    margin-bottom: 18px;
}

.result-panel {
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 18px;
}

.result-buy {
    background: linear-gradient(120deg, #103b35, #142c32);
    border: 1px solid #238b73;
}

.result-no {
    background: linear-gradient(120deg, #362536, #251d2e);
    border: 1px solid #85506d;
}

.result-panel h2 {
    color: white;
    font-size: 25px;
}

.result-panel p {
    color: #c0ccdc;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(100deg, #3976ef, #536df2);
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 44px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #285fd1;
    color: white;
}

[data-testid="stMetric"] {
    background: #111d2e;
    border: 1px solid #293b54;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricLabel"] {
    color: #9aabc3;
}

[data-testid="stMetricValue"] {
    color: white;
}

hr {
    border-color: #26354b;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD DATASET AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")


@st.cache_resource
def load_model():
    with open("purchase_prediction_model.pkl", "rb") as file:
        return pickle.load(file)


try:
    df = load_data()
    model = load_model()

except Exception as e:
    st.error("Unable to load dataset or model.")
    st.write("Please check your project folder and file names.")
    st.code(str(e))
    st.stop()


# Check model features

if not hasattr(model, "feature_names_in_"):
    st.error(
        "Model feature names are missing. "
        "Please save the model after training it with a Pandas DataFrame."
    )
    st.stop()


# =========================================================
# 4. COMMON HERO SECTION
# =========================================================

def show_hero(title, description):

    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">SHOPSENSE AI | MACHINE LEARNING PROJECT</div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div class="brand">ShopSense <span>AI</span></div>
<p style="color:#8293ac;font-size:11px;">
ONLINE SHOPPING ANALYTICS
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Purchase Prediction",
        "Dataset Analysis",
        "Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "**College Mini Project**"
)

st.sidebar.caption(
    "Python | Machine Learning | Streamlit"
)


# =========================================================
# 6. DASHBOARD
# =========================================================

if page == "Dashboard":

    show_hero(
        "Online Shopping Purchase Prediction",
        "Understand customer browsing behaviour and predict "
        "whether a visitor is likely to purchase a product."
    )

    total_sessions = len(df)

    purchase_sessions = int(df["Revenue"].sum())

    no_purchase_sessions = (
        total_sessions - purchase_sessions
    )

    purchase_rate = (
        purchase_sessions / total_sessions
    ) * 100

    # KPI Cards

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("TOTAL SESSIONS", f"{total_sessions:,}"),
        ("PURCHASE SESSIONS", f"{purchase_sessions:,}"),
        ("NO PURCHASE", f"{no_purchase_sessions:,}"),
        ("PURCHASE RATE", f"{purchase_rate:.2f}%")
    ]

    for col, (label, value) in zip(
        [col1, col2, col3, col4],
        metrics
    ):

        with col:

            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Shopping Overview</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    # Purchase Distribution

    with left:

        st.markdown('<div class="info-panel">',
                    unsafe_allow_html=True)

        st.subheader("Purchase Distribution")

        purchase_data = pd.DataFrame({
            "Status": ["Purchase", "No Purchase"],
            "Sessions": [
                purchase_sessions,
                no_purchase_sessions
            ]
        })

        fig = px.pie(
            purchase_data,
            names="Status",
            values="Sessions",
            hole=0.45,
            color_discrete_sequence=[
                "#5b8cff",
                "#42d6cf"
            ]
        )

        fig.update_layout(
            paper_bgcolor="#111d2e",
            plot_bgcolor="#111d2e",
            font_color="white",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Monthly Visitors

    with right:

        st.markdown('<div class="info-panel">',
                    unsafe_allow_html=True)

        st.subheader("Monthly Visitors")

        monthly_data = (
            df.groupby("Month")
            .size()
            .reset_index(name="Visitors")
        )

        fig = px.bar(
            monthly_data,
            x="Month",
            y="Visitors",
            color_discrete_sequence=["#5b8cff"]
        )

        fig.update_layout(
            paper_bgcolor="#111d2e",
            plot_bgcolor="#111d2e",
            font_color="white",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("About This Project")

    st.write(
        "ShopSense AI is a machine learning project that "
        "uses customer website activity to predict purchase intention. "
        "It analyzes browsing pages, time spent, visitor type, "
        "traffic source and other session details."
    )


# =========================================================
# 7. PURCHASE PREDICTION
# =========================================================

elif page == "Purchase Prediction":

    show_hero(
        "Customer Purchase Prediction",
        "Enter website browsing details to predict the "
        "customer's purchase intention."
    )

    with st.form("prediction_form"):

        st.subheader("Customer Browsing Information")

        col1, col2 = st.columns(2)

        with col1:

            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                value=0
            )

            administrative_duration = st.number_input(
                "Administrative Duration",
                min_value=0.0,
                value=0.0
            )

            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                value=0
            )

            informational_duration = st.number_input(
                "Informational Duration",
                min_value=0.0,
                value=0.0
            )

            product_related = st.number_input(
                "Product Related Pages",
                min_value=0,
                value=1
            )

            product_duration = st.number_input(
                "Product Related Duration",
                min_value=0.0,
                value=0.0
            )

            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )

            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )

            page_values = st.number_input(
                "Page Values",
                min_value=0.0,
                value=0.0
            )

        with col2:

            special_day = st.number_input(
                "Special Day",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )

            month = st.selectbox(
                "Month",
                sorted(df["Month"].dropna().unique())
            )

            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].unique())
            )

            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].unique())
            )

            region = st.selectbox(
                "Visitor Region",
                sorted(df["Region"].dropna().unique()),
                format_func=lambda x: f"Region {int(x)}"
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                sorted(df["TrafficType"].unique())
            )

            visitor_type = st.selectbox(
                "Visitor Type",
                sorted(df["VisitorType"].dropna().unique())
            )

            weekend = st.selectbox(
                "Weekend Visit",
                [False, True],
                format_func=lambda x: "Yes" if x else "No"
            )

        submitted = st.form_submit_button(
            "Predict Purchase",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend
        }])

        input_encoded = pd.get_dummies(
            input_data,
            columns=["Month", "VisitorType"],
            drop_first=True,
            dtype=int
        )

        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        try:

            prediction = model.predict(input_encoded)[0]

            probabilities = model.predict_proba(input_encoded)[0]

            purchase_index = list(model.classes_).index(1)

            probability = probabilities[purchase_index] * 100

            if prediction == 1:

                st.markdown(f"""
                <div class="result-panel result-buy">
                    <h2>Purchase Likely</h2>
                    <p>The customer is predicted to make a purchase.</p>
                    <h2>{probability:.2f}%</h2>
                    <p>Purchase Probability</p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-panel result-no">
                    <h2>Purchase Unlikely</h2>
                    <p>The customer is predicted not to make a purchase.</p>
                    <h2>{probability:.2f}%</h2>
                    <p>Purchase Probability</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:

            st.error("Prediction failed. Please check the model.")
            st.code(str(e))


# =========================================================
# 8. DATASET ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    show_hero(
        "Dataset Analysis",
        "Explore the online shopping dataset and understand "
        "different customer browsing patterns."
    )

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{df.shape[0]:,}")

    col2.metric("Total Columns", df.shape[1])

    col3.metric("Duplicate Records", int(df.duplicated().sum()))

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Visitor Type Distribution")

        visitor_data = (
            df["VisitorType"]
            .value_counts()
            .reset_index()
        )

        visitor_data.columns = [
            "Visitor Type",
            "Visitors"
        ]

        fig = px.bar(
            visitor_data,
            x="Visitor Type",
            y="Visitors",
            color_discrete_sequence=["#5b8cff"]
        )

        fig.update_layout(
            paper_bgcolor="#080d18",
            plot_bgcolor="#080d18",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Purchase Rate by Visitor Type")

        visitor_purchase = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .reset_index()
        )

        visitor_purchase["Revenue"] *= 100

        fig = px.bar(
            visitor_purchase,
            x="VisitorType",
            y="Revenue",
            color_discrete_sequence=["#42d6cf"],
            labels={
                "VisitorType": "Visitor Type",
                "Revenue": "Purchase Rate (%)"
            }
        )

        fig.update_layout(
            paper_bgcolor="#080d18",
            plot_bgcolor="#080d18",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Missing Values")

    missing_data = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing_data,
        use_container_width=True
    )


# =========================================================
# 9. MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    show_hero(
        "Machine Learning Model Insights",
        "Understand the performance and important features "
        "of the Random Forest classification model."
    )

    # Prepare dataset

    X = df.drop("Revenue", axis=1)

    X = pd.get_dummies(
        X,
        columns=["Month", "VisitorType"],
        drop_first=True,
        dtype=int
    )

    y = df["Revenue"].astype(int)

    X = X.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    ) * 100

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        f"{accuracy:.2f}%"
    )

    col2.metric(
        "Training Records",
        f"{len(X_train):,}"
    )

    col3.metric(
        "Testing Records",
        f"{len(X_test):,}"
    )

    st.markdown("---")

    st.subheader("Feature Importance")

    if not hasattr(model, "feature_importances_"):

        st.warning(
            "Feature importance is available for tree-based models "
            "such as Random Forest."
        )

    else:

        importance_data = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        })

        importance_data = importance_data.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            importance_data,
            x="Importance",
            y="Feature",
            orientation="h",
            color_discrete_sequence=["#5b8cff"]
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="#080d18",
            plot_bgcolor="#080d18",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Algorithm Information")

    st.markdown("""
    <div class="info-panel">

    **Algorithm:** Random Forest Classifier

    **Learning Type:** Supervised Learning

    **Problem Type:** Binary Classification

    **Target Variable:** Revenue

    **Class 0:** No Purchase

    **Class 1:** Purchase

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <p style="text-align:center;color:#71839d;font-size:12px;">
    ShopSense AI | Online Shopping Purchase Prediction
    <br>
    College Mini Project
    </p>
    """,
    unsafe_allow_html=True
)
