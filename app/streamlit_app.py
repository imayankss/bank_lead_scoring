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
    # MBD amounts are anonymised and noised, so no currency is shown.
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"


def _short_id(customer_id: str) -> str:
    return str(customer_id)[:12]


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

    history = st.sidebar.radio("Product history", ["All clients", "Took a product before", "Never took a product"])
    took = data["products_taken_before"].gt(0)
    keep = {"All clients": took | ~took, "Took a product before": took, "Never took a product": ~took}[history]

    return data[
        data["lead_category"].isin(selected_categories)
        & data["lead_score"].between(selected_score[0], selected_score[1])
        & keep
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

    product_counts = pd.DataFrame(
        {"product": [f"Product {k}" for k in range(1, 5)],
         "clients": [int(data[f"target_{k}_before"].gt(0).sum()) for k in range(1, 5)]}
    )
    fig = px.bar(
        product_counts,
        x="product",
        y="clients",
        color="clients",
        color_continuous_scale="Tealgrn",
        title="Clients Who Already Took Each Product (MBD products are unnamed)",
    )
    st.plotly_chart(fig, width="stretch")


def _render_leads(data: pd.DataFrame) -> None:
    view = data.sort_values(["lead_score", "expected_value"], ascending=False).copy()
    view["client"] = view["customer_id"].map(_short_id)
    table = view[
        [
            "client",
            "recency_days",
            "txn_count_90",
            "products_taken_before",
            "lead_category",
            "lead_score",
            "predicted_propensity",
            "predicted_cltv",
            "expected_value",
        ]
    ].rename(
        columns={
            "client": "Client",
            "recency_days": "Days Since Last Txn",
            "txn_count_90": "Txns (90d)",
            "products_taken_before": "Products Taken",
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
    st.dataframe(table.head(1000), width="stretch", height=460, hide_index=True)  # top 1,000; download has all
    st.download_button(
        "Download filtered leads",
        data=view.to_csv(index=False),
        file_name="filtered_lead_scores.csv",
        mime="text/csv",
    )


def _render_customer_detail(data: pd.DataFrame, explanations: pd.DataFrame) -> None:
    options = data.sort_values("lead_score", ascending=False).head(500)  # a 100k-option picker freezes the browser
    if options.empty:
        st.info("No customers match the active filters.")
        return
    labels = {f"{_short_id(row.customer_id)} | {row.lead_score} ({row.lead_category})": row.customer_id
              for row in options.itertuples()}
    customer_id = labels[st.selectbox("Client (top 500 by score)", list(labels))]
    customer = data.loc[data["customer_id"] == customer_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lead score", f"{customer['lead_score']}/100")
    col2.metric("Category", customer["lead_category"])
    col3.metric("Propensity", f"{customer['predicted_propensity']:.1%}")
    col4.metric("Expected value", _format_currency(customer["expected_value"]))

    activity, products = st.columns(2)
    with activity:
        st.subheader("Activity at scoring date")
        st.write(
            {
                "Days since last transaction": int(customer["recency_days"]),
                "Transactions, last 365 days": int(customer["txn_count_365"]),
                "Transactions, last 90 days": int(customer["txn_count_90"]),
                "Value, last 365 days": _format_currency(customer["total_txn_amt_365"]),
                "Follow-up flags": customer.get("fallback_reasons", "N/A"),
            }
        )
    with products:
        st.subheader("Product history")
        st.write({f"Product {k}: months taken": int(customer[f"target_{k}_before"]) for k in range(1, 5)})

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
    col3.metric("Propensity AUC (95% CI)", f"{row['propensity_auc']:.3f}",
                f"{row['propensity_auc_ci_lower']:.3f} to {row['propensity_auc_ci_upper']:.3f}", delta_color="off")
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
    st.caption("Built for a Bank of India-style lead workflow; data is Sber AI Lab's public MBD-mini "
               "(business clients, CC BY 4.0), not Bank of India customers.")
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
