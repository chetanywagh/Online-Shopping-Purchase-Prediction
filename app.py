
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ShopSense AI | Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)

# =====================================================
# 2. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
    color: #17243b;
}

[data-testid="stSidebar"] {
    background-color: #10213b;
}

[data-testid="stSidebar"] * {
    color: #ffffff;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c5d2e5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(110deg, #142b4d, #24558b);
    padding: 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero p {
    color: #d8e7f8;
    font-size: 14px;
}

.eyebrow {
    color: #9ecbff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #172d4b;
    margin-top: 25px;
    margin-bottom: 15px;
}

.section-description {
    color: #718096;
    font-size: 13px;
    margin-top: -10px;
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    border: 1px solid #e3e9f2;
    border-radius: 13px;
    padding: 19px;
    min-height: 120px;
    box-shadow: 0 3px 12px rgba(25, 50, 85, 0.04);
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #17375e;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
}

.metric-note {
    color: #8b9bb0;
    font-size: 11px;
    margin-top: 4px;
}

.result-card {
    padding: 28px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.purchase {
    background: #e8f7f0;
    border: 1px solid #a6dfc1;
}

.no-purchase {
    background: #fff0f0;
    border: 1px solid #f0b8b8;
}

.result-card h2 {
    font-size: 27px;
    font-weight: 800;
}

.purchase h2 {
    color: #147448;
}

.no-purchase h2 {
    color: #b33d48;
}

.result-card p {
    color: #5c6b7c;
    font-size: 14px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #2059a5;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 45px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #17477f;
    color: white;
    border: none;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3e9f2;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricLabel"] {
    color: #718096;
}

[data-testid="stMetricValue"] {
    color: #17375e;
}

hr {
    border-color: #e2e8f0;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATASET AND TRAINED MODEL
# =====================================================

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

except FileNotFoundError:
    st.error(
        "Dataset or model file not found. "
        "Please upload online_shoppers_intention.csv "
        "and purchase_prediction_model.pkl to your repository."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# =====================================================
# 4. DATA PREPROCESSING
# =====================================================

if "Revenue" not in df.columns:
    st.error("The Revenue target column is missing.")
    st.stop()

df["Revenue"] = df["Revenue"].astype(int)

X = df.drop(columns=["Revenue"])
y = df["Revenue"]

# Same encoding method used in the notebook
X = pd.get_dummies(
    X,
    columns=["Month", "VisitorType"],
    drop_first=True,
    dtype=int
)

# Use exact feature names saved in the trained model
if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    feature_columns = list(X.columns)

if set(feature_columns) != set(X.columns):
    st.error(
        "Model features do not match the dataset. "
        "Please check the notebook preprocessing and saved model."
    )
    st.stop()

X = X.reindex(columns=feature_columns, fill_value=0)

# Calculate test accuracy using the same split settings
try:
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred) * 100

except Exception:
    model_accuracy = None


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown(
    "<h2 style='color:white;'>🛍️ ShopSense AI</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("Online Shopping Intelligence")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Purchase Prediction",
        "Dataset Analysis",
        "Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Project Overview")
st.sidebar.caption(
    "A Machine Learning application that predicts "
    "online customer purchase intention using "
    "browsing behaviour."
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed using Python & Machine Learning")


# =====================================================
# 6. COMMON DATASET METRICS
# =====================================================

total_sessions = len(df)
purchase_sessions = int(df["Revenue"].sum())
no_purchase_sessions = total_sessions - purchase_sessions

purchase_rate = (
    purchase_sessions / total_sessions * 100
    if total_sessions else 0
)


def show_hero(title, subtitle, section):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{section}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def show_metric(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 7. DASHBOARD
# =====================================================

if page == "Dashboard":

    show_hero(
        "Online Shopping Purchase Prediction",
        "Analyze customer browsing behaviour and understand "
        "purchase intention using Machine Learning.",
        "COMMERCE INTELLIGENCE DASHBOARD"
    )

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_metric(
            "Total Sessions",
            f"{total_sessions:,}",
            "Total website visits"
        )

    with c2:
        show_metric(
            "Purchase Sessions",
            f"{purchase_sessions:,}",
            "Sessions ending in purchase"
        )

    with c3:
        show_metric(
            "No Purchase Sessions",
            f"{no_purchase_sessions:,}",
            "Sessions without purchase"
        )

    with c4:
        show_metric(
            "Purchase Rate",
            f"{purchase_rate:.2f}%",
            "Share of sessions with purchase"
        )

    st.markdown(
        '<div class="section-title">Customer Behaviour Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        revenue_counts = pd.DataFrame({
            "Customer Activity": ["Purchase", "No Purchase"],
            "Sessions": [
                purchase_sessions,
                no_purchase_sessions
            ]
        })

        fig = px.pie(
            revenue_counts,
            names="Customer Activity",
            values="Sessions",
            hole=0.62,
            color="Customer Activity",
            color_discrete_map={
                "Purchase": "#2d8a68",
                "No Purchase": "#4b78b9"
            },
            title="Purchase Distribution"
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#24364f",
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        visitor_counts = (
            df["VisitorType"]
            .value_counts()
            .reset_index()
        )
        visitor_counts.columns = ["Visitor Type", "Sessions"]

        fig = px.bar(
            visitor_counts,
            x="Visitor Type",
            y="Sessions",
            color="Visitor Type",
            title="Visitor Type Distribution",
            color_discrete_sequence=[
                "#2059a5", "#4b86c6", "#8bb9e5"
            ]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            font_color="#24364f"
        )

        st.plotly_chart(fig, use_container_width=True)

    if model_accuracy is not None:
        st.info(
            f"Random Forest model test accuracy: "
            f"{model_accuracy:.2f}%"
        )

    st.caption(
        "Dashboard statistics are calculated from the uploaded dataset."
    )


# =====================================================
# 8. PURCHASE PREDICTION
# =====================================================

elif page == "Purchase Prediction":

    show_hero(
        "Customer Purchase Prediction",
        "Enter the browsing details of a website visitor "
        "to predict their purchase intention.",
        "MACHINE LEARNING PREDICTION"
    )

    st.markdown(
        '<div class="section-title">Visitor Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Provide the customer and browsing session details below.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("purchase_prediction_form"):

        st.markdown("#### Customer Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            month = st.selectbox(
                "Month",
                sorted(df["Month"].dropna().unique())
            )

        with c2:
            visitor_type = st.selectbox(
                "Visitor Type",
                sorted(df["VisitorType"].dropna().unique())
            )

        with c3:
            weekend = st.selectbox(
                "Weekend Visit",
                ["No", "Yes"]
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].dropna().unique())
            )

        with c2:
            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].dropna().unique())
            )

        with c3:
            region = st.selectbox(
                "Region",
                sorted(df["Region"].dropna().unique())
            )

        traffic_type = st.selectbox(
            "Traffic Type",
            sorted(df["TrafficType"].dropna().unique())
        )

        st.markdown("---")
        st.markdown("#### Website Browsing Activity")

        c1, c2 = st.columns(2)

        with c1:
            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
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
                value=0
            )

            informational_duration = st.number_input(
                "Informational Duration (seconds)",
                min_value=0.0,
                value=0.0
            )

        with c2:
            product_related = st.number_input(
                "Product Related Pages",
                min_value=0,
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

        c1, c2 = st.columns(2)

        with c1:
            page_values = st.number_input(
                "Page Value",
                min_value=0.0,
                value=0.0
            )

        with c2:
            special_day = st.slider(
                "Special Day",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "Predict Purchase Intention",
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

        # Keep original dataset categories for correct one-hot encoding
        for column in ["Month", "VisitorType"]:
            categories = sorted(df[column].dropna().unique().tolist())
            input_data[column] = pd.Categorical(
                input_data[column],
                categories=categories
            )

        input_encoded = pd.get_dummies(
            input_data,
            columns=["Month", "VisitorType"],
            drop_first=True,
            dtype=int
        )

        input_encoded = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        try:
            prediction = model.predict(input_encoded)[0]
            probabilities = model.predict_proba(input_encoded)[0]

            classes = list(model.classes_)
            purchase_probability = (
                probabilities[classes.index(1)] * 100
            )
            no_purchase_probability = (
                probabilities[classes.index(0)] * 100
            )

            st.markdown("---")
            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:
                st.markdown(f"""
                <div class="result-card purchase">
                    <div style="font-size:42px;">🛒</div>
                    <h2>Purchase Likely</h2>
                    <p>
                    The model predicts that this visitor is likely
                    to make a purchase.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card no-purchase">
                    <div style="font-size:42px;">🛍️</div>
                    <h2>No Purchase Predicted</h2>
                    <p>
                    The model predicts that this visitor is unlikely
                    to make a purchase.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Prediction Probabilities")

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
                "These are model-estimated probabilities, "
                "not guarantees of actual customer behaviour."
            )

        except Exception as e:
            st.error(f"Prediction could not be completed: {e}")


# =====================================================
# 9. DATASET ANALYSIS
# =====================================================

elif page == "Dataset Analysis":

    show_hero(
        "Dataset Analysis",
        "Explore the dataset, understand customer sessions "
        "and analyze important shopping behaviour patterns.",
        "EXPLORATORY DATA ANALYSIS"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### Dataset Preview")
    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.markdown("### Monthly Visitor Sessions")

    monthly = (
        df.groupby("Month")
        .size()
        .reset_index(name="Sessions")
    )

    fig = px.bar(
        monthly,
        x="Month",
        y="Sessions",
        title="Website Sessions by Month",
        color_discrete_sequence=["#3478bd"]
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#24364f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Purchase Behaviour by Visitor Type")

    visitor_revenue = (
        df.groupby("VisitorType")["Revenue"]
        .mean()
        .mul(100)
        .reset_index(name="Purchase Rate")
    )

    fig = px.bar(
        visitor_revenue,
        x="VisitorType",
        y="Purchase Rate",
        title="Purchase Rate by Visitor Type",
        color_discrete_sequence=["#2d8a68"]
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#24364f",
        yaxis_title="Purchase Rate (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Information")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        }),
        use_container_width=True
    )


# =====================================================
# 10. MODEL INSIGHTS
# =====================================================

elif page == "Model Insights":

    show_hero(
        "Model Insights",
        "Understand the trained Random Forest model "
        "and explore the features influencing its predictions.",
        "MACHINE LEARNING INTERPRETATION"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Random Forest")
    c2.metric("Input Features", len(feature_columns))

    if model_accuracy is not None:
        c3.metric("Test Accuracy", f"{model_accuracy:.2f}%")
    else:
        c3.metric("Test Accuracy", "Unavailable")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        st.markdown("### Top Influential Features")

        fig = px.bar(
            importance.head(12).sort_values(
                "Importance",
                ascending=True
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance",
            color_discrete_sequence=["#3478bd"]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="#24364f",
            yaxis_title="",
            xaxis_title="Importance Score"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Feature Importance Details")

        st.dataframe(
            importance,
            use_container_width=True
        )

    st.markdown("### Project Methodology")

    st.markdown("""
    1. Dataset Collection
    2. Data Cleaning and Preprocessing
    3. Exploratory Data Analysis
    4. Categorical Feature Encoding
    5. Train-Test Split
    6. Logistic Regression and Random Forest
    7. Model Evaluation
    8. Purchase Intention Prediction
    """)


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8291a5; padding:12px;">
        <b style="color:#234b78;">ShopSense AI</b><br>
        Online Shopping Purchase Prediction<br>
        <small>Built with Python · Pandas · Scikit-learn · Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
