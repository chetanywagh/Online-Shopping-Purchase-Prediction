
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ShopSense AI",
    page_icon="🛍️",
    layout="wide"
)

# =====================================================
# 2. LOAD DATA AND MODEL
# =====================================================

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
    st.error("Dataset or model file not found. Please check your project files.")
    st.stop()

# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

[data-testid="stSidebar"] {
    background-color: #14243b;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #172b4d;
    margin-bottom: 3px;
}

.sub-title {
    color: #718096;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #172b4d;
    margin-top: 15px;
    margin-bottom: 15px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 13px;
    border: 1px solid #e5eaf2;
    box-shadow: 0px 3px 10px rgba(20, 36, 59, 0.04);
}

.metric-label {
    color: #718096;
    font-size: 14px;
}

.metric-value {
    color: #172b4d;
    font-size: 27px;
    font-weight: 750;
    margin-top: 7px;
}

.info-box {
    background: #eaf2ff;
    border-left: 4px solid #3478f6;
    padding: 15px;
    border-radius: 8px;
    color: #24436c;
    margin-bottom: 15px;
}

.result-box {
    background: white;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #e5eaf2;
    text-align: center;
    margin-top: 15px;
}

div.stButton > button {
    background-color: #2868dc;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #174da8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 🛍️ ShopSense AI")
    st.caption("Online Shopping Intelligence")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Purchase Prediction",
            "Dataset Analysis",
            "Model Insights"
        ]
    )

    st.markdown("---")

    st.markdown("### About Project")
    st.caption(
        "An ML-based application that predicts "
        "whether an online visitor is likely to make a purchase."
    )

    st.markdown("---")
    st.caption("Machine Learning Mini Project")

# =====================================================
# 5. COMMON DATA
# =====================================================

total_visitors = len(df)
total_purchases = int(df["Revenue"].sum())
non_purchases = total_visitors - total_purchases

purchase_rate = (
    total_purchases / total_visitors * 100
    if total_visitors > 0 else 0
)

# =====================================================
# 6. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Shopping Behaviour Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Understand visitor activity and purchasing patterns.</div>',
        unsafe_allow_html=True
    )

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Total Visitors</div>
            <div class="metric-value">{total_visitors:,}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Total Purchases</div>
            <div class="metric-value">{total_purchases:,}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">No Purchases</div>
            <div class="metric-value">{non_purchases:,}</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Purchase Rate</div>
            <div class="metric-value">{purchase_rate:.2f}%</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("")

    left, right = st.columns(2)

    with left:
        st.markdown(
            '<div class="section-title">Purchase Distribution</div>',
            unsafe_allow_html=True
        )

        purchase_data = pd.DataFrame({
            "Status": ["Purchased", "Not Purchased"],
            "Visitors": [total_purchases, non_purchases]
        })

        fig = px.pie(
            purchase_data,
            names="Status",
            values="Visitors",
            hole=0.55,
            color="Status",
            color_discrete_map={
                "Purchased": "#3478f6",
                "Not Purchased": "#aab8cc"
            }
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=20, b=20, l=20, r=20),
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section-title">Monthly Visitor Activity</div>',
            unsafe_allow_html=True
        )

        monthly_data = df.groupby("Month").size().reset_index(name="Visitors")

        month_order = [
            "Feb", "Mar", "May", "June",
            "Jul", "Aug", "Sep", "Oct",
            "Nov", "Dec"
        ]

        monthly_data["Month"] = pd.Categorical(
            monthly_data["Month"],
            categories=month_order,
            ordered=True
        )

        monthly_data = monthly_data.sort_values("Month")

        fig = px.bar(
            monthly_data,
            x="Month",
            y="Visitors",
            color_discrete_sequence=["#3478f6"]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="Month",
            yaxis_title="Number of Visitors"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Project Overview")

    st.markdown(
        """<div class="info-box">
        <b>ShopSense AI</b> uses customer browsing behaviour
        and a Random Forest machine learning model to predict
        whether a visitor is likely to purchase a product.
        </div>""",
        unsafe_allow_html=True
    )

# =====================================================
# 7. PURCHASE PREDICTION
# =====================================================

elif page == "Purchase Prediction":

    st.markdown(
        '<div class="main-title">Purchase Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Enter visitor details to predict their purchase behaviour.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-box">Enter the browsing details of a visitor. The trained ML model will predict the purchase possibility.</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        st.markdown("### 1. Website Browsing Activity")

        col1, col2, col3 = st.columns(3)

        with col1:
            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                max_value=50,
                value=2
            )

            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                max_value=50,
                value=1
            )

        with col2:
            administrative_duration = st.number_input(
                "Administrative Duration (seconds)",
                min_value=0.0,
                value=30.0
            )

            informational_duration = st.number_input(
                "Informational Duration (seconds)",
                min_value=0.0,
                value=20.0
            )

        with col3:
            product_related = st.number_input(
                "Product Related Pages",
                min_value=0,
                max_value=500,
                value=5
            )

            product_duration = st.number_input(
                "Product Page Duration (seconds)",
                min_value=0.0,
                value=120.0
            )

        st.markdown("---")
        st.markdown("### 2. Visitor Behaviour")

        col1, col2, col3 = st.columns(3)

        with col1:
            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.01,
                format="%.3f"
            )

        with col2:
            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.04,
                step=0.01,
                format="%.3f"
            )

        with col3:
            page_values = st.number_input(
                "Page Value",
                min_value=0.0,
                value=5.0
            )

        st.markdown("---")
        st.markdown("### 3. Visitor Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            month = st.selectbox(
                "Visit Month",
                sorted(df["Month"].dropna().unique())
            )

            visitor_type = st.selectbox(
                "Visitor Type",
                sorted(df["VisitorType"].dropna().unique())
            )

        with col2:
            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].dropna().unique())
            )

            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].dropna().unique())
            )

        with col3:
            region = st.selectbox(
                "Region",
                sorted(df["Region"].dropna().unique())
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                sorted(df["TrafficType"].dropna().unique())
            )

        col1, col2 = st.columns(2)

        with col1:
            special_day = st.selectbox(
                "Special Day",
                sorted(df["SpecialDay"].dropna().unique())
            )

        with col2:
            weekend = st.selectbox(
                "Is Weekend?",
                ["No", "Yes"]
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🔍 Predict Purchase Behaviour"
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
            "Weekend": weekend == "Yes"
        }])

        # Keep category order consistent with training data
        month_categories = sorted(df["Month"].dropna().unique())
        visitor_categories = sorted(df["VisitorType"].dropna().unique())

        input_data["Month"] = pd.Categorical(
            input_data["Month"],
            categories=month_categories
        )

        input_data["VisitorType"] = pd.Categorical(
            input_data["VisitorType"],
            categories=visitor_categories
        )

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
        purchase_probability = probabilities[purchase_index] * 100

        st.markdown("---")
        st.markdown("## Prediction Result")

        if prediction == 1:

            st.success("This visitor is predicted to make a purchase.")

            st.markdown(
                f"""<div class="result-box">
                <h2 style="color:#16865b;">Purchase Likely 🛒</h2>
                <h1 style="color:#172b4d;">{purchase_probability:.2f}%</h1>
                <p style="color:#718096;">Predicted purchase probability</p>
                </div>""",
                unsafe_allow_html=True
            )

        else:

            st.warning("This visitor is predicted not to make a purchase.")

            st.markdown(
                f"""<div class="result-box">
                <h2 style="color:#c17b21;">Purchase Not Likely</h2>
                <h1 style="color:#172b4d;">{purchase_probability:.2f}%</h1>
                <p style="color:#718096;">Predicted purchase probability</p>
                </div>""",
                unsafe_allow_html=True
            )

        st.progress(float(purchase_probability / 100))

        st.caption(
            "Note: This is a machine learning prediction, not a guarantee of actual customer behaviour."
        )

# =====================================================
# 8. DATASET ANALYSIS
# =====================================================

elif page == "Dataset Analysis":

    st.markdown(
        '<div class="main-title">Dataset Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Explore the online shopping visitor dataset.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Rows", f"{df.shape[0]:,}")

    with c2:
        st.metric("Total Columns", df.shape[1])

    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("### Purchase Behaviour by Visitor Type")

    visitor_purchase = (
        df.groupby("VisitorType", observed=False)["Revenue"]
        .mean()
        .reset_index()
    )

    visitor_purchase["Purchase Percentage"] = (
        visitor_purchase["Revenue"] * 100
    )

    fig = px.bar(
        visitor_purchase,
        x="VisitorType",
        y="Purchase Percentage",
        color="VisitorType",
        text_auto=".2f",
        color_discrete_sequence=[
            "#3478f6", "#48a9a6", "#9b8de5"
        ]
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        yaxis_title="Purchase Percentage (%)",
        xaxis_title="Visitor Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Visitor Activity")

    fig = px.histogram(
        df,
        x="ProductRelated",
        color="Revenue",
        nbins=30,
        barmode="overlay",
        color_discrete_sequence=["#aab8cc", "#3478f6"],
        labels={
            "Revenue": "Purchase Status",
            "ProductRelated": "Product Related Pages"
        }
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Information")

    st.write("**Column Names:**")
    st.write(list(df.columns))

    st.write("**Statistical Summary:**")
    st.dataframe(df.describe(), use_container_width=True)

# =====================================================
# 9. MODEL INSIGHTS
# =====================================================

elif page == "Model Insights":

    st.markdown(
        '<div class="main-title">Model Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Understand the machine learning model and its important features.</div>',
        unsafe_allow_html=True
    )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"].astype(int)

    X = pd.get_dummies(
        X,
        columns=["Month", "VisitorType"],
        drop_first=True,
        dtype=int
    )

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

    accuracy = accuracy_score(y_test, y_pred) * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Model", "Random Forest")

    with c2:
        st.metric("Model Accuracy", f"{accuracy:.2f}%")

    with c3:
        st.metric("Testing Data", f"{len(X_test):,} Rows")

    st.markdown("### Feature Importance")

    importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Importance Score",
        yaxis_title="Features"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.markdown(
        """<div class="info-box">
        <b>Random Forest:</b> It combines multiple decision trees
        and uses their predictions to make the final prediction.
        Feature importance shows which input features contribute
        more to the model's decisions.
        </div>""",
        unsafe_allow_html=True
    )

# =====================================================
# 10. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8290a5; padding:10px;">
        ShopSense AI | Online Shopping Purchase Prediction
        <br>
        Built using Python, Streamlit and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
