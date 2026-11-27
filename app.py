
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)


# =====================================
# 2. LOAD DATASET AND MODEL
# =====================================

@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")


@st.cache_resource
def load_model():
    with open("purchase_prediction_model.pkl", "rb") as file:
        return pickle.load(file)


df = load_data()
model = load_model()


# =====================================
# 3. PROFESSIONAL DESIGN
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F8FC;
    color: #263445;
}

[data-testid="stSidebar"] {
    background-color: #17365D;
}

[data-testid="stSidebar"] * {
    color: white;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: white;
}

h1, h2, h3 {
    color: #17365D;
    font-weight: 650;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #E0E8F1;
}

[data-testid="stMetricLabel"] {
    color: #64748B;
}

[data-testid="stMetricValue"] {
    color: #17365D;
}

.stButton > button {
    background-color: #2563A6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #17365D;
    color: white;
}

[data-testid="stForm"] {
    background-color: white;
    border: 1px solid #E0E8F1;
    border-radius: 12px;
    padding: 20px;
}

hr {
    border-color: #DCE5EF;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# 4. SIDEBAR NAVIGATION
# =====================================

st.sidebar.title("🛒 ShopSense")

st.sidebar.caption("ONLINE SHOPPING ANALYTICS")

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

st.sidebar.write("Machine Learning Project")

st.sidebar.caption(
    "Developed using Python, Pandas, "
    "Scikit-learn and Streamlit"
)


# =====================================
# 5. DASHBOARD
# =====================================

if page == "Dashboard":

    st.title("Online Shopping Purchase Prediction")

    st.write(
        "Analyze customer shopping behaviour "
        "and predict their purchase intention."
    )

    st.markdown("---")

    total_visitors = len(df)

    purchase_visitors = int(df["Revenue"].sum())

    no_purchase_visitors = (
        total_visitors - purchase_visitors
    )

    purchase_rate = (
        purchase_visitors / total_visitors
    ) * 100

    # Dashboard statistics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Visitors",
        f"{total_visitors:,}"
    )

    col2.metric(
        "Purchase Visitors",
        f"{purchase_visitors:,}"
    )

    col3.metric(
        "No Purchase",
        f"{no_purchase_visitors:,}"
    )

    col4.metric(
        "Purchase Rate",
        f"{purchase_rate:.2f}%"
    )

    st.markdown("---")

    # Purchase distribution

    left, right = st.columns(2)

    with left:

        st.subheader("Purchase Distribution")

        purchase_data = pd.DataFrame({
            "Category": ["Purchase", "No Purchase"],
            "Visitors": [
                purchase_visitors,
                no_purchase_visitors
            ]
        })

        fig = px.pie(
            purchase_data,
            names="Category",
            values="Visitors",
            hole=0.45,
            color_discrete_sequence=[
                "#2563A6",
                "#9DBCE0"
            ]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=25, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("Monthly Visitor Analysis")

        monthly_data = (
            df.groupby("Month")
            .size()
            .reset_index(name="Visitors")
        )

        fig2 = px.bar(
            monthly_data,
            x="Month",
            y="Visitors",
            color_discrete_sequence=["#2563A6"]
        )

        fig2.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=25, b=20),
            xaxis_title="Month",
            yaxis_title="Number of Visitors"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Project Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Rows", f"{df.shape[0]:,}")

    c2.metric("Total Columns", df.shape[1])

    c3.metric("Target Variable", "Revenue")


# =====================================
# 6. PURCHASE PREDICTION
# =====================================

elif page == "Purchase Prediction":

    st.title("Customer Purchase Prediction")

    st.write(
        "Enter customer browsing details "
        "to predict whether the visitor is likely to purchase."
    )

    st.markdown("---")

    st.subheader("Customer Browsing Details")

    with st.form("purchase_form"):

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

            month_options = sorted(
                df["Month"].dropna().unique().tolist()
            )

            month = st.selectbox(
                "Month",
                month_options
            )

            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].unique())
            )

            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].unique())
            )

            region_options = sorted(
                df["Region"].dropna().unique().tolist()
            )

            region = st.selectbox(
                "Visitor Region",
                region_options,
                format_func=lambda x: f"Region {int(x)}"
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                sorted(df["TrafficType"].unique())
            )

            visitor_options = sorted(
                df["VisitorType"].dropna().unique().tolist()
            )

            visitor_type = st.selectbox(
                "Visitor Type",
                visitor_options
            )

            weekend = st.selectbox(
                "Weekend Visit",
                [False, True],
                format_func=lambda x: "Yes" if x else "No"
            )

        submit = st.form_submit_button(
            "Predict Purchase",
            use_container_width=True
        )

    if submit:

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

        prediction = model.predict(input_encoded)[0]

        probabilities = model.predict_proba(input_encoded)[0]

        purchase_index = list(model.classes_).index(1)

        purchase_probability = (
            probabilities[purchase_index] * 100
        )

        st.markdown("---")

        st.subheader("Prediction Result")

        result1, result2 = st.columns(2)

        with result1:

            if prediction == 1:

                st.success(
                    "Customer is likely to make a purchase!"
                )

            else:

                st.warning(
                    "Customer is unlikely to make a purchase."
                )

        with result2:

            st.metric(
                "Purchase Probability",
                f"{purchase_probability:.2f}%"
            )

            st.progress(
                float(purchase_probability / 100)
            )


# =====================================
# 7. DATASET ANALYSIS
# =====================================

elif page == "Dataset Analysis":

    st.title("Dataset Analysis")

    st.write(
        "Explore the dataset and understand "
        "different customer behaviour patterns."
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        st.subheader("Visitor Type Analysis")

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
            color_discrete_sequence=["#2563A6"]
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

        fig2 = px.bar(
            visitor_purchase,
            x="VisitorType",
            y="Revenue",
            color_discrete_sequence=["#7DA9D8"],
            labels={
                "VisitorType": "Visitor Type",
                "Revenue": "Purchase Rate (%)"
            }
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Missing Values Analysis")

    missing_data = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_data.index,
        "Missing Values": missing_data.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# =====================================
# 8. MODEL INSIGHTS
# =====================================

elif page == "Model Insights":

    st.title("Machine Learning Model Insights")

    st.write(
        "Understand the performance of the "
        "Random Forest classification model."
    )

    st.markdown("---")

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

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test Accuracy",
        f"{accuracy:.2f}%"
    )

    c2.metric(
        "Training Samples",
        f"{len(X_train):,}"
    )

    c3.metric(
        "Testing Samples",
        f"{len(X_test):,}"
    )

    st.markdown("---")

    st.subheader("Feature Importance")

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
        color_discrete_sequence=["#2563A6"],
        title="Top 10 Important Features"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Learning Type:** Supervised Learning")

    st.write("**Problem Type:** Binary Classification")

    st.write("**Target Column:** Revenue")

    st.write("**Class 0:** No Purchase")

    st.write("**Class 1:** Purchase")

    st.info(
        "The model learns patterns from previous "
        "customer sessions and predicts purchase intention."
    )


# =====================================
# 9. FOOTER
# =====================================

st.markdown("---")

st.caption(
    "ShopSense | Online Shopping Purchase Prediction"
)
