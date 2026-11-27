import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ShopSense | Purchase Prediction",
    page_icon="🛍️",
    layout="wide"
)


# ==========================================
# 2. LOAD DATASET AND MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")


@st.cache_resource
def load_model():
    with open("purchase_prediction_model.pkl", "rb") as file:
        return pickle.load(file)


df = load_data()
model = load_model()


# ==========================================
# 3. CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F7F7F1;
    color: #26352C;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #183D30;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF;
}

/* Main headings */

h1, h2, h3 {
    color: #183D30;
    font-weight: 650;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 19px;
    border-radius: 12px;
    border: 1px solid #E3E9E0;
}

[data-testid="stMetricLabel"] {
    color: #66776B;
}

[data-testid="stMetricValue"] {
    color: #183D30;
}

/* Buttons */

.stButton > button {
    background-color: #276749;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #183D30;
    color: white;
}

/* Input fields */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] {
    border-radius: 7px;
}

/* Divider */

hr {
    border-color: #DFE7DC;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.markdown("# 🛍️ ShopSense")

st.sidebar.caption("CUSTOMER BEHAVIOUR ANALYTICS")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Purchase Prediction",
        "Customer Analytics",
        "Model Insights"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### About Project")

st.sidebar.write(
    "A Machine Learning application "
    "to understand customer shopping "
    "behaviour and predict purchase intention."
)

st.sidebar.markdown("---")

st.sidebar.caption("Developed using Python & Machine Learning")


# ==========================================
# 5. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.title("Shopping Overview")

    st.write(
        "Understand website visitors, shopping "
        "activities and purchase behaviour."
    )

    st.markdown("---")

    total_visitors = len(df)

    purchase_visitors = int(df["Revenue"].sum())

    non_purchase = total_visitors - purchase_visitors

    purchase_rate = (
        purchase_visitors / total_visitors
    ) * 100

    # Statistics

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sessions", f"{total_visitors:,}")

    c2.metric("Purchases", f"{purchase_visitors:,}")

    c3.metric("No Purchases", f"{non_purchase:,}")

    c4.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    st.markdown("---")

    # Charts

    left, right = st.columns(2)

    with left:

        st.subheader("Purchase Distribution")

        purchase_df = pd.DataFrame({
            "Session Type": ["Purchase", "No Purchase"],
            "Visitors": [purchase_visitors, non_purchase]
        })

        fig = px.pie(
            purchase_df,
            names="Session Type",
            values="Visitors",
            hole=0.55,
            color_discrete_sequence=["#276749", "#E8A75B"]
        )

        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(t=20, b=20, l=20, r=20),
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Monthly Website Visits")

        monthly = (
            df.groupby("Month")
            .size()
            .reset_index(name="Sessions")
        )

        fig2 = px.bar(
            monthly,
            x="Month",
            y="Sessions",
            color_discrete_sequence=["#276749"]
        )

        fig2.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            margin=dict(t=20, b=20),
            xaxis_title="Month",
            yaxis_title="Total Sessions"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Quick Dataset Information")

    a, b, c = st.columns(3)

    a.metric("Dataset Rows", f"{df.shape[0]:,}")

    b.metric("Dataset Columns", df.shape[1])

    c.metric("Target Variable", "Revenue")


# ==========================================
# 6. PURCHASE PREDICTION PAGE
# ==========================================

elif page == "Purchase Prediction":

    st.title("Purchase Prediction")

    st.write(
        "Enter the visitor's website activity "
        "to predict their purchase intention."
    )

    st.markdown("---")

    st.subheader("01. Page Browsing Activity")

    col1, col2, col3 = st.columns(3)

    with col1:

        administrative = st.number_input(
            "Administrative Pages",
            min_value=0,
            value=0
        )

        informational = st.number_input(
            "Informational Pages",
            min_value=0,
            value=0
        )

    with col2:

        administrative_duration = st.number_input(
            "Administrative Duration",
            min_value=0.0,
            value=0.0
        )

        informational_duration = st.number_input(
            "Informational Duration",
            min_value=0.0,
            value=0.0
        )

    with col3:

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

    st.markdown("---")

    st.subheader("02. Website Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:

        bounce_rates = st.number_input(
            "Bounce Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )

    with col2:

        exit_rates = st.number_input(
            "Exit Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01
        )

    with col3:

        page_values = st.number_input(
            "Page Values",
            min_value=0.0,
            value=0.0
        )

    st.markdown("---")

    st.subheader("03. Visitor Information")

    col1, col2, col3 = st.columns(3)

    month_options = sorted(
        df["Month"].dropna().unique().tolist()
    )

    visitor_options = sorted(
        df["VisitorType"].dropna().unique().tolist()
    )

    with col1:

        month = st.selectbox(
            "Month",
            month_options
        )

        special_day = st.number_input(
            "Special Day",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )

    with col2:

        visitor_type = st.selectbox(
            "Visitor Type",
            visitor_options
        )

        region_options = sorted(
            df["Region"].dropna().unique().tolist()
        )

        region = st.selectbox(
            "Visitor Region",
            region_options,
            format_func=lambda x: f"Region {int(x)}"
        )

    with col3:

        operating_system = st.selectbox(
            "Operating System",
            sorted(df["OperatingSystems"].unique())
        )

        browser = st.selectbox(
            "Browser",
            sorted(df["Browser"].unique())
        )

    col1, col2 = st.columns(2)

    with col1:

        traffic_type = st.selectbox(
            "Traffic Type",
            sorted(df["TrafficType"].unique())
        )

    with col2:

        weekend = st.selectbox(
            "Weekend Visit",
            [False, True],
            format_func=lambda x: "Yes" if x else "No"
        )

    st.markdown("---")

    # Prediction

    if st.button("Predict Customer Behaviour", use_container_width=True):

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

        # Apply the same categorical encoding
        input_encoded = pd.get_dummies(
            input_data,
            columns=["Month", "VisitorType"],
            drop_first=True,
            dtype=int
        )

        # Match the original training columns
        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Predict
        prediction = model.predict(input_encoded)[0]

        probabilities = model.predict_proba(input_encoded)[0]

        purchase_index = list(model.classes_).index(1)

        purchase_probability = (
            probabilities[purchase_index] * 100
        )

        st.markdown("### Prediction Result")

        result1, result2 = st.columns(2)

        with result1:

            if prediction == 1:

                st.success(
                    "Purchase Likely\n\n"
                    "The model predicts that this visitor "
                    "is likely to make a purchase."
                )

            else:

                st.warning(
                    "Purchase Unlikely\n\n"
                    "The model predicts that this visitor "
                    "is unlikely to make a purchase."
                )

        with result2:

            st.metric(
                "Purchase Probability",
                f"{purchase_probability:.2f}%"
            )

            st.progress(
                float(purchase_probability / 100)
            )


# ==========================================
# 7. CUSTOMER ANALYTICS PAGE
# ==========================================

elif page == "Customer Analytics":

    st.title("Customer Analytics")

    st.write(
        "Explore different patterns in customer "
        "visits and shopping behaviour."
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Visitor Type Distribution")

        visitor_data = (
            df["VisitorType"]
            .value_counts()
            .reset_index()
        )

        visitor_data.columns = ["Visitor Type", "Sessions"]

        fig = px.bar(
            visitor_data,
            x="Visitor Type",
            y="Sessions",
            color_discrete_sequence=["#276749"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Purchase Rate by Visitor Type")

        visitor_purchase = (
            df.groupby("VisitorType")["Revenue"]
            .mean()
            .reset_index()
        )

        visitor_purchase["Revenue"] *= 100

        fig2 = px.bar(
            visitor_purchase,
            x="VisitorType",
            y="Revenue",
            color_discrete_sequence=["#E8A75B"],
            labels={
                "VisitorType": "Visitor Type",
                "Revenue": "Purchase Rate (%)"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Missing Values Analysis")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# ==========================================
# 8. MODEL INSIGHTS PAGE
# ==========================================

elif page == "Model Insights":

    st.title("Machine Learning Insights")

    st.write(
        "Understand the performance of the "
        "Random Forest classification model."
    )

    st.markdown("---")

    # Prepare features
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

    # Same test split as notebook
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

    c1, c2, c3 = st.columns(3)

    c1.metric("Test Accuracy", f"{accuracy:.2f}%")

    c2.metric("Training Samples", f"{len(X_train):,}")

    c3.metric("Testing Samples", f"{len(X_test):,}")

    st.markdown("---")

    st.subheader("Important Features")

    importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#276749"],
        title="Top 10 Features Influencing Prediction"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Model Details")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Learning Type:** Supervised Learning")

    st.write("**Problem:** Binary Classification")

    st.write("**Target Column:** Revenue")

    st.write("**Class 0:** No Purchase")

    st.write("**Class 1:** Purchase")

    st.info(
        "The model learns customer behaviour from "
        "previous website sessions and predicts "
        "whether a visitor is likely to purchase."
    )


# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "ShopSense | Online Shopping Purchase Prediction "
    "| Machine Learning Project"
)
