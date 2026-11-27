
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Online Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

# =====================================================
# 2. LOAD DATASET AND MODEL
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

except Exception:
    st.error("Please check your dataset and model files.")
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
    background-color: #172b4d;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    color: #172b4d;
    font-size: 32px;
    font-weight: 800;
}

.sub-title {
    color: #718096;
    font-size: 15px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5eaf2;
    box-shadow: 0 3px 10px rgba(20,36,59,0.04);
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
    margin: 12px 0;
}

.result-card {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 14px;
    padding: 25px;
    text-align: center;
    min-height: 145px;
}

.result-number {
    font-size: 34px;
    font-weight: 800;
    margin-top: 12px;
}

.result-label {
    color: #718096;
    font-size: 15px;
}

.final-result {
    background: #eaf2ff;
    border: 1px solid #c9dcff;
    border-radius: 12px;
    padding: 22px;
    text-align: center;
    margin-top: 20px;
}

div.stButton > button {
    background-color: #2868dc;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
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

    st.markdown("# 🛒 Online Purchase")
    st.markdown("### Prediction System")

    st.markdown("---")

    page = st.radio(
        "MAIN MENU",
        [
            "Dashboard",
            "Purchase Prediction",
            "Dataset Analysis",
            "Model Performance"
        ]
    )

    st.markdown("---")

    st.markdown("### Project Details")

    st.caption(
        "Machine Learning based prediction "
        "of online customer purchase behaviour."
    )

    st.markdown("---")
    st.caption("Python | Random Forest | Streamlit")

# =====================================================
# 5. COMMON DATA
# =====================================================

total_visitors = len(df)

total_purchases = int(df["Revenue"].sum())

no_purchases = total_visitors - total_purchases

purchase_rate = (
    total_purchases / total_visitors * 100
    if total_visitors > 0 else 0
)

# =====================================================
# 6. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Customer Visit Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Overview of online visitors and their purchase activity.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("Total Visitors", f"{total_visitors:,}"),
        ("Total Purchases", f"{total_purchases:,}"),
        ("No Purchase", f"{no_purchases:,}"),
        ("Purchase Rate", f"{purchase_rate:.2f}%")
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], cards):

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("")

    left, right = st.columns(2)

    with left:

        st.markdown("### Purchase Distribution")

        purchase_data = pd.DataFrame({
            "Purchase Status": ["Purchased", "Not Purchased"],
            "Visitors": [total_purchases, no_purchases]
        })

        fig = px.pie(
            purchase_data,
            names="Purchase Status",
            values="Visitors",
            hole=0.55,
            color="Purchase Status",
            color_discrete_map={
                "Purchased": "#3478f6",
                "Not Purchased": "#b8c5d6"
            }
        )

        fig.update_layout(
            paper_bgcolor="white",
            margin=dict(t=20, b=20, l=20, r=20),
            legend_title_text=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("### Monthly Purchase Activity")

        monthly = (
            df.groupby("Month", observed=False)["Revenue"]
            .sum()
            .reset_index(name="Purchases")
        )

        month_order = [
            "Feb", "Mar", "May", "June", "Jul",
            "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        monthly["Month"] = pd.Categorical(
            monthly["Month"],
            categories=month_order,
            ordered=True
        )

        monthly = monthly.sort_values("Month")

        fig = px.bar(
            monthly,
            x="Month",
            y="Purchases",
            color_discrete_sequence=["#3478f6"]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="Month",
            yaxis_title="Number of Purchases"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Project Overview")

    st.markdown(
        """
        <div class="info-box">
        This project analyses online customer browsing activity
        and uses a Random Forest model to predict whether a visitor
        is likely to make a purchase.
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# 7. PURCHASE PREDICTION
# =====================================================

elif page == "Purchase Prediction":

    st.markdown(
        '<div class="main-title">Customer Purchase Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Enter customer visit details to predict their purchase decision.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        Enter the customer's browsing activity and visit details.
        The machine learning model will estimate the possibility
        of a purchase.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        st.markdown("### 1. Website Browsing Activity")

        c1, c2, c3 = st.columns(3)

        with c1:

            administrative = st.number_input(
                "Administrative Pages",
                min_value=0,
                value=2
            )

            administrative_duration = st.number_input(
                "Administrative Time (seconds)",
                min_value=0.0,
                value=30.0
            )

        with c2:

            informational = st.number_input(
                "Informational Pages",
                min_value=0,
                value=1
            )

            informational_duration = st.number_input(
                "Informational Time (seconds)",
                min_value=0.0,
                value=20.0
            )

        with c3:

            product_related = st.number_input(
                "Product Related Pages",
                min_value=0,
                value=5
            )

            product_duration = st.number_input(
                "Product Page Time (seconds)",
                min_value=0.0,
                value=120.0
            )

        st.markdown("---")
        st.markdown("### 2. Customer Behaviour")

        c1, c2, c3 = st.columns(3)

        with c1:

            bounce_rates = st.number_input(
                "Bounce Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.01,
                format="%.3f"
            )

        with c2:

            exit_rates = st.number_input(
                "Exit Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.04,
                step=0.01,
                format="%.3f"
            )

        with c3:

            page_values = st.number_input(
                "Page Value",
                min_value=0.0,
                value=5.0
            )

        st.markdown("---")
        st.markdown("### 3. Visit Information")

        c1, c2, c3 = st.columns(3)

        with c1:

            month = st.selectbox(
                "Visit Month",
                sorted(df["Month"].dropna().unique())
            )

            visitor_type = st.selectbox(
                "Visitor Type",
                sorted(df["VisitorType"].dropna().unique())
            )

        with c2:

            operating_system = st.selectbox(
                "Operating System",
                sorted(df["OperatingSystems"].dropna().unique())
            )

            browser = st.selectbox(
                "Browser",
                sorted(df["Browser"].dropna().unique())
            )

        with c3:

            region = st.selectbox(
                "Region",
                sorted(df["Region"].dropna().unique()),
                format_func=lambda x: f"Region {x}"
            )

            traffic_type = st.selectbox(
                "Traffic Type",
                sorted(df["TrafficType"].dropna().unique())
            )

        c1, c2 = st.columns(2)

        with c1:

            special_day = st.selectbox(
                "Special Day",
                sorted(df["SpecialDay"].dropna().unique())
            )

        with c2:

            weekend = st.selectbox(
                "Weekend Visit",
                ["No", "Yes"]
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "🔍 Predict Customer Purchase",
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
            "Weekend": weekend == "Yes"
        }])

        # Match training category order

        month_categories = sorted(
            df["Month"].dropna().unique()
        )

        visitor_categories = sorted(
            df["VisitorType"].dropna().unique()
        )

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

        # MODEL PREDICTION

        prediction = model.predict(input_encoded)[0]

        probabilities = model.predict_proba(input_encoded)[0]

        purchase_index = list(model.classes_).index(1)
        no_purchase_index = list(model.classes_).index(0)

        purchase_probability = (
            probabilities[purchase_index] * 100
        )

        no_purchase_probability = (
            probabilities[no_purchase_index] * 100
        )

        # FINAL CUSTOMER DECISION

        st.markdown("---")
        st.markdown("## Prediction Result")

        if prediction == 1:

            st.markdown(
                """
                <div class="final-result">
                    <h2 style="color:#16865b;">
                        ✓ Customer Will Make a Purchase
                    </h2>
                    <p style="color:#52647c;">
                        The model predicts that this customer
                        is likely to complete a purchase.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="final-result">
                    <h2 style="color:#c17b21;">
                        Customer Will Not Make a Purchase
                    </h2>
                    <p style="color:#52647c;">
                        The model predicts that this customer
                        is unlikely to complete a purchase.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ONLY TWO PROBABILITY CARDS

        st.markdown("### Purchase Probability")

        p1, p2 = st.columns(2)

        with p1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Probability of Purchase
                    </div>
                    <div class="result-number" style="color:#16865b;">
                        {purchase_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Probability of No Purchase
                    </div>
                    <div class="result-number" style="color:#c17b21;">
                        {no_purchase_probability:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "The two percentages represent the model's estimated "
            "probabilities for purchase and no purchase. "
            "Together, they add up to 100%."
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
        '<div class="sub-title">Explore the dataset, statistical summary and customer patterns.</div>',
        unsafe_allow_html=True
    )

    # DATASET OVERVIEW

    st.markdown("### Dataset Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Records", f"{df.shape[0]:,}")

    with c2:
        st.metric("Total Columns", df.shape[1])

    with c3:
        st.metric("Purchase Records", f"{total_purchases:,}")

    # DATASET PREVIEW

    st.markdown("---")
    st.markdown("### Dataset Preview")

    st.caption("Sample records from the original dataset.")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=350
    )

    # STATISTICAL SUMMARY

    st.markdown("---")
    st.markdown("### Statistical Summary")

    st.caption(
        "Numerical summary showing count, mean, standard deviation, "
        "minimum, maximum and quartile values."
    )

    numerical_summary = df.describe().T

    st.dataframe(
        numerical_summary.round(2),
        use_container_width=True
    )

    # PURCHASE BY VISITOR TYPE

    st.markdown("---")
    st.markdown("### Purchase Rate by Visitor Type")

    visitor_data = (
        df.groupby("VisitorType", observed=False)["Revenue"]
        .mean()
        .mul(100)
        .reset_index(name="Purchase Rate")
    )

    fig = px.bar(
        visitor_data,
        x="VisitorType",
        y="Purchase Rate",
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
        xaxis_title="Visitor Type",
        yaxis_title="Purchase Rate (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # PRODUCT PAGE VISITS

    st.markdown("### Product Page Visits")

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
        plot_bgcolor="white",
        xaxis_title="Number of Product Pages",
        yaxis_title="Number of Visitors"
    )

    st.plotly_chart(fig, use_container_width=True)

    # WEEKEND AND WEEKDAY PURCHASES

    st.markdown("### Weekend vs Weekday Purchases")

    weekend_data = (
        df.groupby("Weekend", observed=False)["Revenue"]
        .mean()
        .mul(100)
        .reset_index(name="Purchase Rate")
    )

    weekend_data["Day Type"] = weekend_data["Weekend"].map({
        True: "Weekend",
        False: "Weekday"
    })

    fig = px.bar(
        weekend_data,
        x="Day Type",
        y="Purchase Rate",
        color="Day Type",
        text_auto=".2f",
        color_discrete_sequence=["#3478f6", "#48a9a6"]
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        xaxis_title="Visit Type",
        yaxis_title="Purchase Rate (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 9. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Evaluate the trained Random Forest model.</div>',
        unsafe_allow_html=True
    )

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
        st.metric("Algorithm", "Random Forest")

    with c2:
        st.metric("Model Accuracy", f"{accuracy:.2f}%")

    with c3:
        st.metric("Testing Records", f"{len(X_test):,}")

    # FEATURE IMPORTANCE

    st.markdown("---")
    st.markdown("### Important Features")

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
        yaxis_title="Feature"
    )

    st.plotly_chart(fig, use_container_width=True)

    # CLASSIFICATION REPORT

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
        """
        <div class="info-box">
        <b>Random Forest:</b> It combines multiple decision trees
        to make a final prediction. Feature importance helps us
        understand which input features contribute more to
        the model's predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# 10. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8290a5; padding:12px;">
        <b>Online Purchase Prediction</b>
        <br>
        Developed using Python, Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
