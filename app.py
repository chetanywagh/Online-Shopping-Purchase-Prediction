import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    page_icon="🛒",
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
# 3. SIMPLE CSS DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #17365D;
}

[data-testid="stSidebar"] * {
    color: white;
}

h1, h2, h3 {
    color: #17365D;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #DCE5EF;
}

.stButton > button {
    background-color: #2563A6;
    color: white;
    border-radius: 7px;
    border: none;
}

.stButton > button:hover {
    background-color: #17365D;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR
# ==========================================

st.sidebar.title("🛒 Shopping Prediction")

st.sidebar.write("College Mini Project")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Purchase Prediction",
        "Dataset Analysis",
        "Model Information"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed using Python and Machine Learning"
)


# ==========================================
# 5. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("Online Shopping Purchase Prediction")

    st.write(
        "This project predicts whether a customer "
        "will purchase a product based on their "
        "website browsing behaviour."
    )

    st.markdown("---")

    total_visitors = len(df)

    purchase = int(df["Revenue"].sum())

    no_purchase = total_visitors - purchase

    purchase_rate = (purchase / total_visitors) * 100

    # Display statistics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sessions", f"{total_visitors:,}")

    col2.metric("Purchase Sessions", f"{purchase:,}")

    col3.metric("No Purchase", f"{no_purchase:,}")

    col4.metric("Purchase Rate", f"{purchase_rate:.2f}%")

    st.markdown("---")

    # Purchase chart

    left, right = st.columns(2)

    with left:

        st.subheader("Purchase Distribution")

        purchase_data = pd.DataFrame({
            "Purchase Status": ["Purchase", "No Purchase"],
            "Visitors": [purchase, no_purchase]
        })

        fig = px.pie(
            purchase_data,
            names="Purchase Status",
            values="Visitors",
            hole=0.4,
            color_discrete_sequence=["#2563A6", "#9DBCE0"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.subheader("Monthly Visitors")

        month_data = (
            df.groupby("Month")
            .size()
            .reset_index(name="Visitors")
        )

        fig = px.bar(
            month_data,
            x="Month",
            y="Visitors",
            color_discrete_sequence=["#2563A6"]
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("About the Project")

    st.write(
        "The system uses customer session information "
        "such as page visits, browsing duration, "
        "visitor type and weekend activity to predict "
        "purchase intention."
    )


# ==========================================
# 6. PURCHASE PREDICTION
# ==========================================

elif page == "Purchase Prediction":

    st.title("Customer Purchase Prediction")

    st.write(
        "Enter the customer browsing details "
        "to predict the purchase result."
    )

    st.markdown("---")

    with st.form("prediction_form"):

        st.subheader("Customer Details")

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

        # Convert categorical columns into numerical columns

        input_encoded = pd.get_dummies(
            input_data,
            columns=["Month", "VisitorType"],
            drop_first=True,
            dtype=int
        )

        # Match training dataset columns

        input_encoded = input_encoded.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Make prediction

        prediction = model.predict(input_encoded)[0]

        probabilities = model.predict_proba(input_encoded)[0]

        purchase_index = list(model.classes_).index(1)

        probability = probabilities[purchase_index] * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.success(
                    "Customer is likely to make a purchase."
                )

            else:

                st.warning(
                    "Customer is unlikely to make a purchase."
                )

        with col2:

            st.metric(
                "Purchase Probability",
                f"{probability:.2f}%"
            )

            st.progress(float(probability / 100))


# ==========================================
# 7. DATASET ANALYSIS
# ==========================================

elif page == "Dataset Analysis":

    st.title("Dataset Analysis")

    st.write(
        "Explore the dataset and understand "
        "customer shopping behaviour."
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

        st.subheader("Visitor Type")

        visitor_data = (
            df["VisitorType"]
            .value_counts()
            .reset_index()
        )

        visitor_data.columns = [
            "Visitor Type",
            "Count"
        ]

        fig = px.bar(
            visitor_data,
            x="Visitor Type",
            y="Count",
            color_discrete_sequence=["#2563A6"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Purchase by Visitor Type")

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
            color_discrete_sequence=["#7DA9D8"],
            labels={
                "VisitorType": "Visitor Type",
                "Revenue": "Purchase Rate (%)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

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


# ==========================================
# 8. MODEL INFORMATION
# ==========================================

elif page == "Model Information":

    st.title("Model Information")

    st.write(
        "This section shows the machine learning "
        "algorithm and its performance."
    )

    st.markdown("---")

    # Prepare data for evaluation

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
        "Model Accuracy",
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

    st.subheader("Top Important Features")

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
        color_discrete_sequence=["#2563A6"]
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Algorithm Used")

    st.write("**Random Forest Classifier**")

    st.write("**Learning Type:** Supervised Learning")

    st.write("**Problem Type:** Binary Classification")

    st.write("**Target Column:** Revenue")

    st.write("**0:** No Purchase")

    st.write("**1:** Purchase")


# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Online Shopping Purchase Prediction | College Mini Project"
)
