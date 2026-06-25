"""Streamlit dashboard for the lead scoring project."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from lead_scoring.dashboard_data import (
    load_dashboard_data,
    load_decile_lift,
    load_explanations,
    load_feature_importance,
    load_metrics,
)


st.set_page_config(
    page_title="Lead Scoring Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"INR {value:,.0f}"


def _mask_email(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    if "@" not in text:
        return "N/A"
    name, domain = text.split("@", 1)
    return f"{name[:2]}***@{domain}"


def _mask_phone(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return f"******{text[-4:]}" if len(text) >= 4 else "N/A"


@st.cache_data
def _load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_dashboard_data(),
        load_metrics(),
        load_decile_lift(),
        load_feature_importance(),
        load_explanations(),
    )


def _sidebar_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    categories = sorted(data["lead_category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("Lead category", categories, default=categories)

    min_score = int(data["lead_score"].min())
    max_score = int(data["lead_score"].max())
    selected_score = st.sidebar.slider("Lead score", min_score, max_score, (min_score, max_score))

    segments = sorted(data["Customer_Segment"].dropna().unique().tolist())
    selected_segments = st.sidebar.multiselect("Customer segment", segments, default=segments)

    products = sorted(data["Best_First_Option"].dropna().unique().tolist())
    selected_products = st.sidebar.multiselect("Primary product", products, default=products)

    return data[
        data["lead_category"].isin(selected_categories)
        & data["lead_score"].between(selected_score[0], selected_score[1])
        & data["Customer_Segment"].isin(selected_segments)
        & data["Best_First_Option"].isin(selected_products)
    ].copy()


def _render_kpis(data: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Scored customers", f"{len(data):,}")
    col2.metric("Hot leads", f"{(data['lead_category'] == 'Hot').sum():,}")
    col3.metric("Medium leads", f"{(data['lead_category'] == 'Medium').sum():,}")
    col4.metric("Average score", f"{data['lead_score'].mean():.1f}")
    col5.metric("Expected value", _format_currency(data["expected_value"].sum()))


def _render_overview(data: pd.DataFrame) -> None:
    _render_kpis(data)
    left, right = st.columns(2)
    with left:
        counts = data["lead_category"].value_counts().reset_index()
        counts.columns = ["lead_category", "count"]
        fig = px.bar(
            counts,
            x="lead_category",
            y="count",
            color="lead_category",
            color_discrete_map={"Hot": "#dc2626", "Medium": "#d97706", "Cold": "#2563eb"},
            title="Lead Category Distribution",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(
            data,
            x="lead_score",
            nbins=20,
            color_discrete_sequence=["#334155"],
            title="Lead Score Distribution",
        )
        st.plotly_chart(fig, width="stretch")

    product_counts = data["Best_First_Option"].value_counts().head(10).reset_index()
    product_counts.columns = ["product", "leads"]
    fig = px.bar(
        product_counts,
        x="product",
        y="leads",
        color="leads",
        color_continuous_scale="Tealgrn",
        title="Top Primary Product Recommendations",
    )
    st.plotly_chart(fig, width="stretch")


def _render_leads(data: pd.DataFrame) -> None:
    view = data.sort_values(["lead_score", "expected_value"], ascending=False).copy()
    table = view[
        [
            "Customer_ID",
            "First_Name",
            "Last_Name",
            "Customer_Segment",
            "Best_First_Option",
            "lead_category",
            "lead_score",
            "predicted_propensity",
            "predicted_cltv",
            "expected_value",
        ]
    ].rename(
        columns={
            "Customer_ID": "Customer ID",
            "First_Name": "First Name",
            "Last_Name": "Last Name",
            "Customer_Segment": "Segment",
            "Best_First_Option": "Primary Product",
            "lead_category": "Category",
            "lead_score": "Score",
            "predicted_propensity": "Propensity",
            "predicted_cltv": "Predicted CLTV",
            "expected_value": "Expected Value",
        }
    )
    table["Propensity"] = table["Propensity"].map(lambda x: f"{x:.1%}")
    table["Predicted CLTV"] = table["Predicted CLTV"].map(_format_currency)
    table["Expected Value"] = table["Expected Value"].map(_format_currency)
    st.dataframe(table, width="stretch", height=460, hide_index=True)
    st.download_button(
        "Download filtered leads",
        data=view.to_csv(index=False),
        file_name="filtered_lead_scores.csv",
        mime="text/csv",
    )


def _render_customer_detail(data: pd.DataFrame, explanations: pd.DataFrame) -> None:
    options = data.sort_values("lead_score", ascending=False)
    labels = [
        f"{row.Customer_ID} | {row.First_Name} {row.Last_Name} | {row.lead_score} ({row.lead_category})"
        for row in options.itertuples()
    ]
    if not labels:
        st.info("No customers match the active filters.")
        return
    selected = st.selectbox("Customer", labels)
    customer_id = selected.split(" | ", 1)[0]
    customer = data.loc[data["Customer_ID"] == customer_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lead score", f"{customer['lead_score']}/100")
    col2.metric("Category", customer["lead_category"])
    col3.metric("Propensity", f"{customer['predicted_propensity']:.1%}")
    col4.metric("Expected value", _format_currency(customer["expected_value"]))

    profile, products = st.columns(2)
    with profile:
        st.subheader("Profile")
        st.write(
            {
                "Name": f"{customer['First_Name']} {customer['Last_Name']}",
                "Segment": customer["Customer_Segment"],
                "Occupation": customer["Occupation"],
                "Annual income": _format_currency(customer["Annual_Income"]),
                "Risk score": int(customer["Risk_Score"]),
                "Email": _mask_email(customer["Email"]),
                "Phone": _mask_phone(customer["Phone"]),
            }
        )
    with products:
        st.subheader("Product alignment")
        st.write(
            {
                "Primary recommendation": customer["Best_First_Option"],
                "Secondary recommendation": customer["Best_Second_Option"],
                "Chosen product": customer["Chosen_Product"],
                "Matches primary": customer["Chosen_Product"] == customer["Best_First_Option"],
                "Matches secondary": customer["Chosen_Product"] == customer["Best_Second_Option"],
            }
        )

    if not explanations.empty:
        match = explanations.loc[explanations["customer_id"] == customer_id]
        if not match.empty:
            st.subheader("Top model drivers")
            st.write(match.iloc[0].get("top_drivers", "Not available"))


def _render_model_quality(metrics: pd.DataFrame, lift: pd.DataFrame) -> None:
    if metrics.empty:
        st.warning("Metrics are not available. Run the pipeline first.")
        return
    row = metrics.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test rows", f"{int(row['rows']):,}")
    col2.metric("Positive rate", f"{row['positive_rate']:.1%}")
    col3.metric("Propensity AUC", f"{row['propensity_auc']:.3f}")
    col4.metric("CLTV RMSE", _format_currency(row["cltv_rmse"]))

    if not lift.empty:
        fig = px.bar(
            lift.sort_values("decile"),
            x="decile",
            y="conversion_rate",
            title="Conversion Rate by Expected Value Decile",
            color="conversion_rate",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(lift, width="stretch", hide_index=True)


def _render_explainability(importance: pd.DataFrame) -> None:
    if importance.empty:
        st.warning("Feature importance is not available. Run the pipeline first.")
        return
    value_col = "importance_mean" if "importance_mean" in importance.columns else "mean_abs_shap"
    top = importance.sort_values(value_col, ascending=False).head(15)
    fig = px.bar(
        top.sort_values(value_col),
        x=value_col,
        y="feature",
        orientation="h",
        title="Top Model Drivers",
        color=value_col,
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(top, width="stretch", hide_index=True)


def main() -> None:
    data, metrics, lift, importance, explanations = _load_all()
    if data.empty:
        st.error("No lead score data found. Run the pipeline first.")
        return

    st.title("Lead Scoring and CLTV Dashboard")
    filtered = _sidebar_filters(data)
    if filtered.empty:
        st.warning("No leads match the active filters.")
        return

    overview, leads, customer, quality, explain = st.tabs(
        ["Overview", "Leads", "Customer Detail", "Model Quality", "Explainability"]
    )
    with overview:
        _render_overview(filtered)
    with leads:
        _render_leads(filtered)
    with customer:
        _render_customer_detail(filtered, explanations)
    with quality:
        _render_model_quality(metrics, lift)
    with explain:
        _render_explainability(importance)


if __name__ == "__main__":
    main()
