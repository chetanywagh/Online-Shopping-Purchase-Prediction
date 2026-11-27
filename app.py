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
    page_title="ShopSense | Smart Shopping",
    page_icon="🛍️",
    layout="wide"
)


# ==========================================
# 2. LOAD DATA AND MODEL
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
# 3. WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F6F8;
    color: #202938;
}

/* Hide default sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* Main content */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Navigation */
.nav-title {
    font-size: 27px;
    font-weight: 800;
    color: #182B49;
}

.nav-title span {
    color: #F28C28;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #182B49, #304E73);
    padding: 48px 42px;
    border-radius: 18px;
    color: white;
    margin-top: 15px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 39px;
    margin-bottom: 12px;
}

.hero p {
    color: #E5EAF1;
    font-size: 16px;
    line-height: 1.7;
}

/* Section heading */
.section-heading {
    font-size: 24px;
    font-weight: 750;
    color: #182B49;
    margin-top: 25px;
    margin-bottom: 12px;
}

/* Cards */
.info-card {
    background: white;
    padding: 23px;
    border-radius: 13px;
    border: 1px solid #E8EAF0;
    min-height: 145px;
}

.info-card h3 {
    color: #182B49;
    font-size: 19px;
}

.info-card p {
    color: #6B7280;
    font-size: 14px;
    line-height: 1.6;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E8EAF0;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricValue"] {
    color: #182B49;
}

/* Buttons */
.stButton > button {
    background-color: #F28C28;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 650;
}

.stButton > button:hover {
    background-color: #D97412;
    color: white;
}

/* Form */
[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border: 1px solid #E8EAF0;
    border-radius: 14px;
}

/* Navigation radio */
div[role="radiogroup"] {
    gap: 10px;
}

div[role="radiogroup"] label {
    background: white;
    border: 1px solid #E3E6EC;
    border-radius: 8px;
    padding: 8px 14px;
}

hr {
    border-color: #E2E5EA;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. TOP NAVIGATION
# ==========================================

top1, top2 = st.columns([1, 3])

with top1:
    st.markdown(
        '<div class="nav-title">Shop<span>Sense.</span></div>',
        unsafe_allow_html=True
    )

with top2:
    page = st.radio(
        "Navigation",
        [
            "Home",
            "Purchase Prediction",
            "Explore Data",
            "About Project"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("---")


# ==========================================
# 5. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Understand Your Customers Better.</h1>
        <p>
            Explore online shopping behaviour and discover
            whether a website visitor is likely to make a purchase.
            <br>
            Powered by Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">Shopping Insights</div>',
        unsafe_allow_html=True
    )

    total = len(df)
    purchases = int(df["Revenue"].sum())
    no_purchase = total - purchases
    rate = purchases / total * 100

    a, b, c, d = st.columns(4)

    a.metric("Total Sessions", f"{total:,}")
    b.metric("Successful Purchases", f"{purchases:,}")
    c.metric("No Purchases", f"{no_purchase:,}")
    d.metric("Purchase Rate", f"{rate:.2f}%")

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Explore ShopSense</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>🛍️ Purchase Prediction</h3>
            <p>
                Enter customer browsing details and
                predict their purchase intention
                using our trained ML model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Customer Analytics</h3>
            <p>
                Explore visitor activity, monthly
                sessions and purchasing patterns
                through interactive charts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Machine Learning</h3>
            <p>
                Understand model performance
                and discover which features
                influence purchase predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">Purchase Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        purchase_data = pd.DataFrame({
            "Result": ["Purchase", "No Purchase"],
            "Sessions": [purchases, no_purchase]
        })

        fig = px.pie(
            purchase_data,
            names="Result",
            values="Sessions",
            hole=0.5,
            color_discrete_sequence=["#F28C28", "#233D60"]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        monthly = (
            df.groupby("Month")
            .size()
            .reset_index(name="Sessions")
        )

        fig2 = px.bar(
            monthly,
            x="Month",
            y="Sessions",
            color_discrete_sequence=["#F28C28"]
        )

        fig2.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(fig2, use_container_width=True)


# ==========================================
# 6. PURCHASE PREDICTION PAGE
# ==========================================

elif page == "Purchase Prediction":

    st.title("🛍️ Customer Purchase Prediction")

    st.write(
        "Fill in the visitor's browsing information "
        "to estimate their purchase intention."
    )

    st.info(
        "Enter the details of one website session below."
    )

    with st.form("prediction_form"):

        st.subheader("1. Browsing Activity")

        c1, c2 = st.columns(2)

        with c1:

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

        with c2:

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

            page_values = st.number_input(
                "Page Values",
                min_value=0.0,
                value=0.0
            )

        st.markdown("---")

        st.subheader("2. Website Behaviour")

        c1, c2 = st.columns(2)

        with c1:

            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )

        with c2:

            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )

        st.markdown("---")

        st.subheader("3. Visitor Information")

        c1, c2, c3 = st.columns(3)

        month_options = sorted(
            df["Month"].dropna().unique().tolist()
        )

        visitor_options = sorted(
            df["VisitorType"].dropna().unique().tolist()
        )

        region_options = sorted(
            df["Region"].dropna().unique().tolist()
        )

        with c1:

            month = st.selectbox(
                "Month",
                month_options
            )

            visitor_type = st.selectbox(
                "Visitor Type",
                visitor_options
            )

            special_day = st.number_input(
                "Special Day",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )

        with c2:

            region = st.selectbox(
                "Visitor Region",
                region_options,
                format_func=lambda x: f"Region {int(x)}"
            )

            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].unique())
            )

        with c3:

            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].unique())
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                sorted(df["TrafficType"].unique())
            )

            weekend = st.selectbox(
                "Weekend Visit",
                [False, True],
                format_func=lambda x: "Yes" if x else "No"
            )

        submit = st.form_submit_button(
            "Predict Purchase Intention",
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

        st.subheader("Your Prediction Result")

        r1, r2 = st.columns(2)

        with r1:

            if prediction == 1:

                st.success(
                    "Purchase Likely! The visitor is predicted "
                    "to make a purchase."
                )

            else:

                st.warning(
                    "Purchase Unlikely! The visitor is predicted "
                    "not to make a purchase."
                )

        with r2:

            st.metric(
                "Purchase Probability",
                f"{purchase_probability:.2f}%"
            )

            st.progress(
                float(purchase_probability / 100)
            )


# ==========================================
# 7. EXPLORE DATA PAGE
# ==========================================

elif page == "Explore Data":

    st.title("Explore Customer Data")

    st.write(
        "Explore the dataset and understand "
        "how visitors interact with the website."
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Visitor Type")

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
            color_discrete_sequence=["#F28C28"]
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        st.subheader("Purchase Rate by Visitor")

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
            color_discrete_sequence=["#233D60"],
            labels={
                "VisitorType": "Visitor Type",
                "Revenue": "Purchase Rate (%)"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

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
# 8. ABOUT PROJECT PAGE
# ==========================================

elif page == "About Project":

    st.title("About ShopSense")

    st.write(
        "ShopSense is an Online Shopping Purchase "
        "Prediction project developed using Machine Learning."
    )

    st.markdown("---")

    st.subheader("Project Objective")

    st.write(
        "The main objective is to analyze customer "
        "browsing behaviour and predict whether a "
        "visitor will make a purchase or not."
    )

    st.subheader("Machine Learning Approach")

    st.write(
        "We use a Random Forest Classifier trained "
        "on historical online shopping session data."
    )

    st.markdown("---")

    st.subheader("Project Information")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        **Dataset**

        - Online Shoppers Intention
        - Target: Revenue
        - Classification problem
        - Purchase = 1
        - No Purchase = 0
        """)

    with c2:

        st.markdown("""
        **Technologies Used**

        - Python
        - Pandas
        - Scikit-learn
        - Plotly
        - Streamlit
        - Random Forest
        """)

    st.markdown("---")

    st.subheader("Project Workflow")

    st.write("""
    1. Dataset Collection
    2. Data Cleaning and Preprocessing
    3. Exploratory Data Analysis
    4. Model Training
    5. Model Evaluation
    6. Purchase Prediction
    """)


# ==========================================
# 9. FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "<center style='color:#777777;'>"
    "© ShopSense | Online Shopping Purchase Prediction"
    "</center>",
    unsafe_allow_html=True
)
