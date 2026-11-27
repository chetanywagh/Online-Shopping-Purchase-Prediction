
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ShopSense AI | Purchase Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080d19;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #263449;
}

.hero {
    background: linear-gradient(120deg, #111f3d, #142d50, #10223b);
    border: 1px solid #29466d;
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #b8c9e2;
    font-size: 16px;
}

.tag {
    display: inline-block;
    background: #173b60;
    color: #76c7ff;
    border: 1px solid #28618c;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #142238, #101a2c);
    border: 1px solid #273d5c;
    padding: 22px;
    border-radius: 17px;
    min-height: 125px;
}

.metric-label {
    color: #9aacc5;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 800;
}

.metric-icon {
    font-size: 22px;
}

.section-heading {
    color: #ffffff;
    font-size: 23px;
    font-weight: 700;
    margin: 20px 0 15px 0;
}

.result-success {
    background: linear-gradient(120deg, #103b35, #102b2c);
    border: 1px solid #1d806d;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
}

.result-danger {
    background: linear-gradient(120deg, #40202b, #2b1b2b);
    border: 1px solid #a34a65;
    padding: 28px;
    border-radius: 18px;
    text-align: center;
}

.result-title {
    color: white;
    font-size: 28px;
    font-weight: 800;
}

.result-subtitle {
    color: #d1dbe8;
    font-size: 14px;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: 700;
    min-height: 48px;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0284c7);
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #273d5c;
    padding: 16px;
    border-radius: 14px;
}

.stTextInput input, .stNumberInput input {
    background: #111d30;
    color: white;
}

hr {
    border-color: #263449;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATASET AND TRAINED MODEL
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv("online_shoppers_intention.csv")


@st.cache_resource
def load_model():
    with open("purchase_prediction_model.pkl", "rb") as file:
        return pickle.load(file)


try:
    df = load_dataset()
    model = load_model()

except FileNotFoundError as e:
    st.error(
        "Required file not found: " + str(e) +
        ". Please upload the dataset CSV and trained PKL file "
        "to your GitHub repository."
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# --------------------------------------------------
# PREPARE DATA AND FEATURES
# --------------------------------------------------

if "Revenue" not in df.columns:
    st.error("Revenue column is missing from the dataset.")
    st.stop()

X = df.drop("Revenue", axis=1)

X = pd.get_dummies(
    X,
    columns=["Month", "VisitorType"],
    drop_first=True,
    dtype=int
)

if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    feature_columns = list(X.columns)

# Check that the dataset and model have matching features
if set(feature_columns) != set(X.columns):
    st.error(
        "Dataset features do not match the saved model. "
        "Please check the preprocessing used in your notebook."
    )
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🛍️ ShopSense AI")
st.sidebar.caption("Intelligent Purchase Prediction")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🔮 Purchase Predictor",
        "📊 Dataset Explorer",
        "🧠 Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### About the Project")
st.sidebar.caption(
    "ShopSense AI uses Machine Learning "
    "to understand online shopping behaviour "
    "and predict customer purchase intention."
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python · Streamlit · Scikit-learn")


# --------------------------------------------------
# COMMON METRICS
# --------------------------------------------------

total_sessions = len(df)
purchase_sessions = int(df["Revenue"].sum())
no_purchase_sessions = total_sessions - purchase_sessions

purchase_rate = (
    purchase_sessions / total_sessions * 100
    if total_sessions > 0 else 0
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="tag">MACHINE LEARNING PROJECT</div>
        <h1>Understand Intent.<br>Predict Purchases.</h1>
        <p>
        An intelligent platform that analyzes online shopping
        behaviour and predicts whether a visitor is likely to
        make a purchase.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Business Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-label">Total Sessions</div>
            <div class="metric-value">{total_sessions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🛒</div>
            <div class="metric-label">Purchase Sessions</div>
            <div class="metric-value">{purchase_sessions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🚶</div>
            <div class="metric-label">No Purchase Sessions</div>
            <div class="metric-value">{no_purchase_sessions:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-label">Purchase Rate</div>
            <div class="metric-value">{purchase_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Purchase Behaviour</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Purchase Distribution")
        chart_data = pd.DataFrame({
            "Session Type": ["Purchase", "No Purchase"],
            "Sessions": [purchase_sessions, no_purchase_sessions]
        }).set_index("Session Type")

        st.bar_chart(chart_data)

    with col2:
        st.markdown("#### Visitor Type")
        visitor_data = df["VisitorType"].value_counts()
        st.bar_chart(visitor_data)

    st.info(
        "Explore the Purchase Predictor to test a visitor's "
        "shopping session using the trained Machine Learning model."
    )


# --------------------------------------------------
# PURCHASE PREDICTOR
# --------------------------------------------------

elif page == "🔮 Purchase Predictor":

    st.markdown("""
    <div class="hero">
        <div class="tag">AI-POWERED PREDICTION</div>
        <h1>Purchase Predictor</h1>
        <p>
        Enter the visitor's browsing activity and let
        Machine Learning predict their purchase intention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">👤 Visitor Information</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            month_options = sorted(df["Month"].dropna().unique().tolist())
            month = st.selectbox("Visit Month", month_options)

            visitor_options = sorted(
                df["VisitorType"].dropna().unique().tolist()
            )
            visitor_type = st.selectbox(
                "Visitor Type", visitor_options
            )

            weekend = st.selectbox(
                "Was the visit on a weekend?",
                ["No", "Yes"]
            )

            special_day = st.slider(
                "Special Day Value",
                0.0, 1.0, 0.0, 0.1
            )

            region = st.selectbox(
                "Region",
                sorted(df["Region"].dropna().unique().tolist())
            )

        with col2:
            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].dropna().unique().tolist())
            )

            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].dropna().unique().tolist())
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                sorted(df["TrafficType"].dropna().unique().tolist())
            )

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">📄 Page Activity</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                max_value=1000,
                value=0
            )

            administrative_duration = st.number_input(
                "Administrative Duration (seconds)",
                min_value=0.0,
                value=0.0
            )

            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                max_value=1000,
                value=0
            )

            informational_duration = st.number_input(
                "Informational Duration (seconds)",
                min_value=0.0,
                value=0.0
            )

        with col2:
            product_related = st.number_input(
                "Product Related Pages",
                min_value=0,
                max_value=1000,
                value=1
            )

            product_related_duration = st.number_input(
                "Product Related Duration (seconds)",
                min_value=0.0,
                value=30.0
            )

            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.01,
                format="%.4f"
            )

            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.04,
                step=0.01,
                format="%.4f"
            )

        page_values = st.number_input(
            "Page Value",
            min_value=0.0,
            value=0.0
        )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🔮 Predict Purchase Intention",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
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
            "Weekend": weekend == "Yes"
        }])

        # Apply the same encoding used during model training
        input_encoded = pd.get_dummies(
            input_data,
            columns=["Month", "VisitorType"],
            drop_first=True,
            dtype=int
        )

        # Keep the exact training feature names and order
        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        try:
            prediction = model.predict(input_encoded)[0]
            probabilities = model.predict_proba(input_encoded)[0]

            classes = list(model.classes_)
            purchase_index = classes.index(1)
            no_purchase_index = classes.index(0)

            purchase_probability = probabilities[purchase_index] * 100
            no_purchase_probability = probabilities[no_purchase_index] * 100

            st.markdown("---")
            st.markdown(
                '<div class="section-heading">Prediction Result</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:

                st.markdown(f"""
                <div class="result-success">
                    <div style="font-size:45px;">🛒</div>
                    <div class="result-title">Purchase Likely!</div>
                    <p class="result-subtitle">
                    The model predicts that this visitor is likely
                    to make a purchase.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="result-danger">
                    <div style="font-size:45px;">🛍️</div>
                    <div class="result-title">No Purchase Predicted</div>
                    <p class="result-subtitle">
                    The model predicts that this visitor is
                    unlikely to make a purchase.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Prediction Confidence")

            p1, p2 = st.columns(2)

            with p1:
                st.metric(
                    "Purchase Probability",
                    f"{purchase_probability:.2f}%"
                )
                st.progress(float(purchase_probability / 100))

            with p2:
                st.metric(
                    "No Purchase Probability",
                    f"{no_purchase_probability:.2f}%"
                )
                st.progress(float(no_purchase_probability / 100))

            st.caption(
                "Probabilities are model estimates, not a guarantee "
                "of actual customer behaviour."
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "📊 Dataset Explorer":

    st.markdown("""
    <div class="hero">
        <div class="tag">DATA ANALYTICS</div>
        <h1>Dataset Explorer</h1>
        <p>Explore the online shopping sessions used in this project.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(df.columns) - 1)
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Revenue Distribution")

    revenue_counts = df["Revenue"].value_counts().rename(
        index={0: "No Purchase", 1: "Purchase"}
    )

    st.bar_chart(revenue_counts)

    st.markdown("### Visitor Type Distribution")
    st.bar_chart(df["VisitorType"].value_counts())

    st.markdown("### Monthly Sessions")
    st.bar_chart(df["Month"].value_counts())


# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------

elif page == "🧠 Model Insights":

    st.markdown("""
    <div class="hero">
        <div class="tag">MODEL INTELLIGENCE</div>
        <h1>Machine Learning Insights</h1>
        <p>
        Understand how the trained Random Forest model
        analyzes online shopping behaviour.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model Information")

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Random Forest")
    c2.metric("Estimators", "100")
    c3.metric("Input Features", len(feature_columns))

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        st.markdown("### Feature Importance")

        st.bar_chart(
            importance_df.head(12).set_index("Feature")
        )

        st.markdown("### Feature Importance Details")
        st.dataframe(
            importance_df,
            use_container_width=True
        )

    st.markdown("### Project Workflow")

    st.markdown("""
    1. Data Collection
    2. Data Cleaning and Preprocessing
    3. Exploratory Data Analysis
    4. Feature Encoding
    5. Model Training
    6. Model Evaluation
    7. Purchase Prediction
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8293aa; padding:15px;">
        <b>ShopSense AI</b> · Online Shopping Purchase Prediction
        <br>
        Built using Python, Pandas, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
