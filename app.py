
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --------------------------------
# 1. PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)


# --------------------------------
# 2. LOAD DATASET AND MODEL
# --------------------------------

df = pd.read_csv("online_shoppers_intention.csv")

with open("purchase_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)


# --------------------------------
# 3. BASIC DESIGN
# --------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background-color: #eaf1fa;
    }

    h1, h2, h3 {
        color: #17365d;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("🛒 ShopPredict")

st.sidebar.write("Online Shopping Analysis")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Purchase Prediction",
        "Dataset Analysis",
        "Model Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Machine Learning Project")
st.sidebar.write("Developed using Python & Streamlit")


# --------------------------------
# 5. DASHBOARD
# --------------------------------

if page == "Dashboard":

    st.title("Online Shopping Purchase Prediction")

    st.write(
        "Analyze customer shopping behaviour "
        "and predict their purchase intention."
    )

    total_visitors = len(df)

    purchase_visitors = int(df["Revenue"].sum())

    no_purchase_visitors = (
        total_visitors - purchase_visitors
    )

    purchase_rate = (
        purchase_visitors / total_visitors
    ) * 100

    # Display important values

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Visitors", total_visitors)

    col2.metric("Purchase Visitors", purchase_visitors)

    col3.metric("No Purchase", no_purchase_visitors)

    col4.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    st.markdown("---")

    st.subheader("Purchase Overview")

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
        title="Purchase vs No Purchase"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Visitor Analysis")

    monthly_data = df.groupby("Month").size().reset_index(
        name="Visitors"
    )

    fig2 = px.bar(
        monthly_data,
        x="Month",
        y="Visitors",
        title="Visitors by Month"
    )

    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------
# 6. PURCHASE PREDICTION
# --------------------------------

elif page == "Purchase Prediction":

    st.title("Customer Purchase Prediction")

    st.write(
        "Enter customer browsing details "
        "to predict whether the customer will make a purchase."
    )

    st.markdown("---")

    st.subheader("Customer Behaviour Details")

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

    st.markdown("---")

    # --------------------------------
    # 7. PREDICTION BUTTON
    # --------------------------------

    if st.button("Predict Purchase", use_container_width=True):

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

        # Keep the same categories used in training

        input_data["Month"] = pd.Categorical(
            input_data["Month"],
            categories=month_options
        )

        input_data["VisitorType"] = pd.Categorical(
            input_data["VisitorType"],
            categories=visitor_options
        )

        # Convert categorical columns into numerical columns

        input_encoded = pd.get_dummies(
            input_data,
            columns=["Month", "VisitorType"],
            drop_first=True,
            dtype=int
        )

        # Match the model's training columns

        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Make prediction

        prediction = model.predict(input_encoded)[0]

        probabilities = model.predict_proba(input_encoded)[0]

        purchase_index = list(model.classes_).index(1)

        purchase_probability = probabilities[purchase_index] * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("Customer is likely to make a purchase!")

        else:

            st.warning("Customer is unlikely to make a purchase.")

        st.metric(
            "Purchase Probability",
            f"{purchase_probability:.2f}%"
        )


# --------------------------------
# 8. DATASET ANALYSIS
# --------------------------------

elif page == "Dataset Analysis":

    st.title("Dataset Analysis")

    st.write(
        "Explore the dataset and understand "
        "different customer behaviour patterns."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    col1.metric("Total Rows", df.shape[0])

    col2.metric("Total Columns", df.shape[1])

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Visitor Type Analysis")

    visitor_data = df.groupby(
        "VisitorType"
    ).size().reset_index(name="Visitors")

    fig = px.bar(
        visitor_data,
        x="VisitorType",
        y="Visitors",
        title="Visitors by Visitor Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Purchase Rate by Visitor Type")

    visitor_purchase = df.groupby(
        "VisitorType"
    )["Revenue"].mean().reset_index()

    visitor_purchase["Revenue"] *= 100

    fig2 = px.bar(
        visitor_purchase,
        x="VisitorType",
        y="Revenue",
        title="Purchase Rate by Visitor Type",
        labels={"Revenue": "Purchase Rate (%)"}
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Missing Values")

    missing_data = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_data.index,
        "Missing Values": missing_data.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


# --------------------------------
# 9. MODEL INSIGHTS
# --------------------------------

elif page == "Model Insights":

    st.title("Machine Learning Model Insights")

    st.write(
        "This project uses Random Forest Classifier "
        "to predict customer purchase behaviour."
    )

    st.markdown("---")

    # Prepare data in the same way as model training

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

    # Same train-test split used during model training

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Calculate test accuracy

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred) * 100

    st.metric(
        "Model Test Accuracy",
        f"{accuracy:.2f}%"
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
        title="Top 10 Important Features"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Problem Type:** Binary Classification")

    st.write("**Target Column:** Revenue")

    st.write("**Prediction Classes:**")

    st.write("- 0 = No Purchase")

    st.write("- 1 = Purchase")

