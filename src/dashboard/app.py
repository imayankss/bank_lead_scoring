# src/dashboard/app.py
from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Ensure project root is on sys.path so `src.*` imports work when run via Streamlit
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dashboard.data_access import (
    load_customer_360_dashboard,
    load_customer_360_summary,
    load_hero_slice,
)

# ---------------------------------------------------------
# Page + modal styling
# ---------------------------------------------------------
st.set_page_config(page_title="Customer 360 — Unified Lead Scoring", layout="wide")

st.markdown(
    """
<style>
/* 16:9 responsive modal box styling */
[data-testid="stDialog"] > div[role="dialog"],
[data-testid="stModal"] > div[role="dialog"] {
  aspect-ratio: 16 / 9;
  width: min(90vw, 1400px);
  height: auto;
  max-width: none !important;
  border-radius: 16px;
  overflow: hidden;
  padding: 0 !important;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
}

/* Inner content fills full area */
[data-testid="stDialog"] > div[role="dialog"] > div,
[data-testid="stModal"] > div[role="dialog"] > div {
  width: 100%;
  height: 100%;
}

/* Scroll if content overflows */
[data-testid="stDialog"] [data-testid="stVerticalBlock"],
[data-testid="stModal"] [data-testid="stVerticalBlock"] {
  max-height: 100%;
  overflow-y: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper formatting
# ---------------------------------------------------------
def _fmt_int(x):
    try:
        return int(x) if pd.notna(x) else 0
    except Exception:
        return 0


def _fmt_f2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "0.00"


@st.cache_data(show_spinner=False)
def load_dashboard_data():
    """
    Load all dashboard data from DuckDB via the data_access helpers.
    Cached so that UI interactions don’t re-query each time.
    """
    df_360 = load_customer_360_dashboard(limit=None)
    df_summary = load_customer_360_summary()
    df_hero = load_hero_slice()
    return df_360, df_summary, df_hero


def render_customer_card(row: pd.Series, full_df: pd.DataFrame):
    st.markdown(f"### {row.get('full_name', 'Customer')} (`{row['cust_id']}`)")
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        # Scores
        st.write(
            f"- **Hybrid score**: "
            f"{_fmt_int(row.get('hybrid_score_0_100_final'))} "
            f"| **ML proba**: "
            f"{float(row.get('ml_unified_customer_proba') or 0):.3f} "
            f"| **CLTV profit (final)**: "
            f"{_fmt_int(row.get('cltv_profit_final'))}"
        )
        st.write(
            f"- **CLTV decile**: {_fmt_int(row.get('cltv_decile_final'))} "
            f"| **Priority segment**: {row.get('priority_segment')} "
            f"| **Action bucket**: {row.get('action_bucket')}"
        )

        # Profile
        st.write(
            f"- **Profile**: {row.get('gender', 'N/A')}, "
            f"age={_fmt_int(row.get('age'))}, "
            f"income={_fmt_f2(row.get('income_annual_inr'))} INR "
            f"({row.get('income_segment', 'N/A')})"
        )
        st.write(
            f"- **Risk / Location**: risk_bucket={row.get('risk_bucket', 'N/A')}, "
            f"region={row.get('region', 'N/A')}, "
            f"zone={row.get('zone', 'N/A')}, "
            f"urban_rural={row.get('urban_rural_flag', 'N/A')}"
        )

        # Products / balances
        st.write(
            f"- **Products**: num_products={_fmt_int(row.get('num_products'))}, "
            f"loan_flag={_fmt_int(row.get('label_has_loan_product'))}, "
            f"any_product={_fmt_int(row.get('label_has_any_product'))}"
        )
        st.write(
            f"- **Balances**: "
            f"CBS total clr bal={_fmt_f2(row.get('cbs_total_clr_bal_amt'))}, "
            f"AA inflows={_fmt_f2(row.get('aa_avg_monthly_inflows'))}, "
            f"AA outflows={_fmt_f2(row.get('aa_avg_monthly_outflows'))}"
        )

        # Lead behavior
        st.write(
            f"- **Leads**: lead_cnt={_fmt_int(row.get('lead_cnt'))}, "
            f"has_any_lead={_fmt_int(row.get('label_has_any_lead'))}, "
            f"last_source={row.get('last_lead_source', 'N/A')}, "
            f"last_date={row.get('last_lead_date', 'N/A')}, "
            f"recency={_fmt_int(row.get('lead_recency_days'))} days "
            f"({row.get('lead_recency_bucket', 'N/A')})"
        )
        st.write(
            f"- **Last product**: last_prod_product_id="
            f"{row.get('last_prod_product_id', 'N/A')}"
        )

        with st.expander("Raw row"):
            st.dataframe(pd.DataFrame([row]).T, use_container_width=True)

    with c2:
        st.markdown("**Download**")
        st.download_button(
            "Download this customer",
            full_df[full_df["cust_id"] == row["cust_id"]].to_csv(index=False).encode(),
            file_name=f"customer_{row['cust_id']}.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
df_360, df_summary, df_hero = load_dashboard_data()

st.title("Customer 360 — Unified Lead Scoring")

# ---------- sidebar filters ----------
st.sidebar.header("Filters")

priority_options = sorted(df_360["priority_segment"].dropna().unique().tolist())
priority_selected = st.sidebar.multiselect(
    "Priority segment", options=priority_options, default=priority_options
)

action_options = sorted(df_360["action_bucket"].dropna().unique().tolist())
action_selected = st.sidebar.multiselect(
    "Action bucket", options=action_options, default=action_options
)

recency_options = sorted(df_360["lead_recency_bucket"].dropna().unique().tolist())
recency_selected = st.sidebar.multiselect(
    "Lead recency bucket", options=recency_options, default=recency_options
)

source_options = sorted(df_360["last_lead_source"].dropna().unique().tolist())
source_selected = st.sidebar.multiselect(
    "Last lead source", options=source_options, default=source_options
)

min_hybrid = st.sidebar.slider("Min hybrid score (0–100)", 0, 100, 0)

q = st.sidebar.text_input("Search (cust_id / full_name / source)", "")

# ---------- apply filters ----------
df_view = df_360.copy()

if priority_selected:
    df_view = df_view[df_view["priority_segment"].isin(priority_selected)]
if action_selected:
    df_view = df_view[df_view["action_bucket"].isin(action_selected)]
if recency_selected:
    df_view = df_view[df_view["lead_recency_bucket"].isin(recency_selected)]
if source_selected:
    df_view = df_view[df_view["last_lead_source"].isin(source_selected)]

df_view = df_view[df_view["hybrid_score_0_100_final"] >= min_hybrid]

if q:
    ql = q.lower()

    def _m(s: pd.Series) -> pd.Series:
        return s.fillna("").str.lower().str.contains(ql, na=False)

    df_view = df_view[
        _m(df_view["cust_id"])
        | _m(df_view["full_name"])
        | _m(df_view["last_lead_source"])
    ]

# ---------- header KPIs ----------
total_customers = len(df_360)
filtered_customers = len(df_view)
high_share = (
    (df_view["priority_segment"] == "HIGH").mean() * 100 if filtered_customers else 0.0
)

cA, cB, cC = st.columns(3)
cA.metric("Total customers", int(total_customers))
cB.metric("Filtered customers", int(filtered_customers))
cC.metric("HIGH share (filtered)", f"{high_share:.1f}%")

# ---------------------------------------------------------
# Tabs: Overview | Segment Summary | Customer Explorer | Hero Outreach
# ---------------------------------------------------------
tab_overview, tab_summary, tab_customers, tab_hero = st.tabs(
    ["Overview", "Segment summary", "Customer explorer", "Hero outreach"]
)

# ---------- Overview tab ----------
with tab_overview:
    st.subheader("Overview (filtered subset)")

    c1, c2 = st.columns(2)
    # Priority distribution
    if filtered_customers:
        priority_counts = (
            df_view["priority_segment"].value_counts().reset_index()
        )
        priority_counts.columns = ["priority_segment", "n"]
        c1.write("Priority mix")
        c1.bar_chart(
            priority_counts.set_index("priority_segment")["n"],
            use_container_width=True,
        )

        # Action bucket distribution
        action_counts = df_view["action_bucket"].value_counts().reset_index()
        action_counts.columns = ["action_bucket", "n"]
        c2.write("Action bucket mix")
        c2.bar_chart(
            action_counts.set_index("action_bucket")["n"],
            use_container_width=True,
        )

    st.write("---")
    st.write("Top 10 customers by hybrid score (filtered)")
    top10 = (
        df_view.sort_values(
            ["hybrid_score_0_100_final", "ml_unified_customer_proba", "cltv_profit_final"],
            ascending=[False, False, False],
        )
        .head(10)
        .loc[
            :,
            [
                "cust_id",
                "full_name",
                "priority_segment",
                "action_bucket",
                "lead_recency_bucket",
                "last_lead_source",
                "hybrid_score_0_100_final",
                "ml_unified_customer_proba",
                "cltv_profit_final",
            ],
        ]
    )
    st.dataframe(top10, use_container_width=True)

# ---------- Segment summary tab ----------
with tab_summary:
    st.subheader("Segment summary (priority × action × recency × source)")
    st.dataframe(df_summary, use_container_width=True)

# ---------- Customer explorer tab ----------
with tab_customers:
    st.subheader("Customer explorer")

    show_cols = [
        "cust_id",
        "full_name",
        "priority_segment",
        "action_bucket",
        "lead_recency_bucket",
        "last_lead_source",
        "hybrid_score_0_100_final",
        "ml_unified_customer_proba",
        "cltv_profit_final",
        "cltv_decile_final",
        "num_products",
        "label_has_loan_product",
        "label_has_any_lead",
    ]
    existing = [c for c in show_cols if c in df_view.columns]
    grid_df = (
        df_view.sort_values(
            ["hybrid_score_0_100_final", "ml_unified_customer_proba", "cltv_profit_final"],
            ascending=[False, False, False],
        )[existing]
        .reset_index(drop=True)
    )

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(animateRows=True, rowHeight=32)
    go = gb.build()

    grid = AgGrid(
        grid_df,
        gridOptions=go,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        height=420,
        theme="streamlit",
    )

    sel = grid.get("selected_rows", [])
    if hasattr(sel, "to_dict"):
        sel = sel.to_dict(orient="records")
    if isinstance(sel, list) and len(sel) > 0:
        clicked = sel[0].get("cust_id")
        if clicked and st.session_state.get("opened_for") != clicked:
            st.session_state["opened_for"] = clicked
            row = df_360[df_360["cust_id"] == clicked]
            if len(row):

                @st.dialog(f"Customer {clicked}")
                def _dlg():
                    render_customer_card(row.iloc[0], df_360)

                _dlg()

    st.write("---")
    st.subheader("Customer detail (manual)")
    options = df_view["cust_id"].tolist()
    cust_pick = st.selectbox(
        "Select customer", options, index=0 if options else None, key="manual_cust"
    )
    if cust_pick:
        render_customer_card(
            df_360[df_360["cust_id"] == cust_pick].iloc[0], df_360
        )

    st.write("---")
    st.download_button(
        "Download filtered customers (CSV)",
        df_view.to_csv(index=False).encode(),
        file_name="customer_360_filtered.csv",
        mime="text/csv",
    )

# ---------- Hero outreach tab ----------
with tab_hero:
    st.subheader("Hero slice — HIGH, A1/A2, recent, digital sources")
    st.caption("This is the pre-computed ‘top outreach list’ you built in DuckDB.")
    st.dataframe(df_hero, use_container_width=True)

    st.download_button(
        "Download hero outreach list (CSV)",
        df_hero.to_csv(index=False).encode(),
        file_name="customer_360_hero_slice_high_recent_digital.csv",
        mime="text/csv",
    )
