# src/dashboard/app.py
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import altair as alt
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
# Page styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CRM Leads Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* filter strip + metric cards */
.crm-filter-row {
  background: #f7f9fc;
  border: 1px solid #e1e7f5;
  padding: 0.8rem 1rem;
  border-radius: 12px;
  margin-bottom: 1rem;
}
.crm-metric-card {
  background: white;
  border-radius: 12px;
  border: 1px solid #eef1f7;
  padding: 1rem;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
}
[data-testid="stDialog"] > div[role="dialog"],
[data-testid="stModal"] > div[role="dialog"] {
  width: min(90vw, 1200px);
  border-radius: 18px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Constants + helpers
# ---------------------------------------------------------
LEAD_BUCKET_ORDER = ["0_30d", "31_60d", "61_90d", "91_365d", "GT_365d"]

# Human-friendly labels for lead recency (age of last lead)
RECENCY_LABEL_MAP = {
    "0_30d": "0–30 days",
    "31_60d": "31–60 days",
    "61_90d": "61–90 days",
    "91_365d": "91–365 days",
    "GT_365d": ">365 days",
    "Unknown": "No recent lead",
}

PRODUCT_NAME_MAP = {
    "P001": "Current / CASA",
    "P002": "Credit Card",
    "P003": "Loan & Lending",
    "P004": "Investment / Wealth",
    "P005": "Savings & Deposits",
}

PRODUCT_CATEGORY_MAP = {
    "P001": "CASA",
    "P002": "Cards",
    "P003": "Lending",
    "P004": "Investment",
    "P005": "Deposit",
}

STATUS_FRIENDLY = {
    "A1_HIGH_LEAD_NO_LOAN_UPSELL": "High priority · Upsell (no existing loan)",
    "A2_HIGH_LEAD_WITH_LOAN_CROSSSELL": "High priority · Cross-sell (has loan)",
    "B1_MEDIUM_WARM": "Medium priority · Warm follow-up",
    "B2_LOW_WARM": "Low priority · Warm / nurture",
    "C1_COLD": "Very low priority · Cold / long-term nurture",
}


def _fmt_int(x) -> int:
    try:
        return int(x) if pd.notna(x) else 0
    except Exception:
        return 0


def _fmt_float(x, digits=2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return f"{0:.{digits}f}"


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_360 = load_customer_360_dashboard(limit=None)
    df_summary = load_customer_360_summary()
    df_hero = load_hero_slice()
    return df_360, df_summary, df_hero


def enrich_customer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cosmetic fields used by the CRM dashboard layout.
    """
    out = df.copy()

    # Geography
    out["center"] = out["region"].fillna("Unknown")
    out["zone"] = out["zone"].fillna("Unknown")
    out["branch"] = out["zone"].fillna("Unknown") + " Hub"

    # Product labelling
    out["product_name"] = (
        out["last_prod_product_id"].map(PRODUCT_NAME_MAP).fillna("Unclassified")
    )
    out["product_category"] = (
        out["last_prod_product_id"].map(PRODUCT_CATEGORY_MAP).fillna("Other")
    )

    # Friendly status label
    out["status_label"] = out["action_bucket"].map(STATUS_FRIENDLY).fillna(
        out["action_bucket"].fillna("Unassigned")
    )

    # Dates
    out["creation_date"] = pd.to_datetime(out["last_lead_date"], errors="coerce")
    out["creation_month"] = out["creation_date"].dt.strftime("%b-%Y").fillna("Unknown")

    # Recency buckets + human label
    out["lead_recency_bucket"] = pd.Categorical(
        out["lead_recency_bucket"].fillna("Unknown"),
        categories=LEAD_BUCKET_ORDER + ["Unknown"],
        ordered=True,
    )
    out["lead_recency_label"] = (
        out["lead_recency_bucket"].astype(str).map(RECENCY_LABEL_MAP).fillna("Unknown")
    )

    # Scores with safe defaults
    out["hybrid_score_0_100_final"] = out["hybrid_score_0_100_final"].fillna(0)
    out["ml_unified_customer_proba"] = out["ml_unified_customer_proba"].fillna(0.0)
    out["cltv_profit_final"] = out["cltv_profit_final"].fillna(0.0)

    return out


def _dropdown_filter(label: str, options: Iterable[str]) -> str:
    """
    Single-select dropdown with 'All' option.
    Returns a string (one of the options or 'All').
    """
    opts = sorted({str(o) for o in options if pd.notna(o)})
    if not opts:
        return st.selectbox(label, ["All"])
    return st.selectbox(label, ["All"] + opts)


def build_filter_controls(df: pd.DataFrame) -> Dict[str, object]:
    """
    Main filter ribbon at the top + advanced filters in sidebar.
    """
    # ---------------------------
    # MAIN FILTER STRIP (top)
    # ---------------------------
    st.markdown("### Lead filters")
    with st.container():
        st.markdown('<div class="crm-filter-row">', unsafe_allow_html=True)

        # Row 1: Center / Zone / Branch / Status
        row1 = st.columns(4)
        with row1[0]:
            center = _dropdown_filter("Center", df["center"].unique())
        with row1[1]:
            zone = _dropdown_filter("Zone", df["zone"].unique())
        with row1[2]:
            branch = _dropdown_filter("Branch", df["branch"].unique())
        with row1[3]:
            status = _dropdown_filter("Status", df["status_label"].unique())

        # Row 2: Product category / Product / Source / Month
        row2 = st.columns(4)
        with row2[0]:
            prod_cat = _dropdown_filter("Product category", df["product_category"].unique())
        with row2[1]:
            product = _dropdown_filter("Product", df["product_name"].unique())
        with row2[2]:
            source = _dropdown_filter("Source", df["last_lead_source"].unique())
        with row2[3]:
            month = _dropdown_filter("Lead month", df["creation_month"].unique())

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # SIDEBAR FILTERS (left)
    # ---------------------------
    with st.sidebar:
        st.markdown("### Advanced lead filters")

        min_score = st.slider("Hybrid score ≥", 0, 100, 0)

        date_min = df["creation_date"].min()
        date_max = df["creation_date"].max()
        default_range = (
            date_min.date() if pd.notna(date_min) else datetime(2020, 1, 1).date(),
            date_max.date() if pd.notna(date_max) else datetime.today().date(),
        )
        date_range = st.date_input(
            "Creation date window",
            value=default_range,
            format="YYYY-MM-DD",
        )

        search = st.text_input(
            "Search cust ID / name / source",
            placeholder="E.g. C00010, Arjun, Campaign",
        )

    return {
        "center": center,
        "zone": zone,
        "branch": branch,
        "status_label": status,
        "product_category": prod_cat,
        "product_name": product,
        "last_lead_source": source,
        "creation_month": month,
        "min_score": min_score,
        "date_range": date_range,
        "search": search,
    }




def apply_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    view = df.copy()

    # Columns controlled by dropdowns
    for col in [
        "center",
        "zone",
        "branch",
        "status_label",
        "product_category",
        "product_name",
        "last_lead_source",
        "creation_month",
    ]:
        selected = filters.get(col)
        if selected and selected != "All":
            view = view[view[col] == selected]

    # Hybrid score threshold
    view = view[view["hybrid_score_0_100_final"] >= filters.get("min_score", 0)]

    # Date range
    date_range = filters.get("date_range")
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        if start and end:
            view = view[
                (view["creation_date"].dt.date >= start)
                & (view["creation_date"].dt.date <= end)
            ]

    # Text search
    search = (filters.get("search") or "").strip().lower()
    if search:
        mask = (
            view["cust_id"].astype(str).str.lower().str.contains(search, na=False)
            | view["full_name"].astype(str).str.lower().str.contains(search, na=False)
            | view["last_lead_source"].astype(str).str.lower().str.contains(search, na=False)
        )
        view = view[mask]

    return view.reset_index(drop=True)



def render_metric_cards(df_full: pd.DataFrame, df_filtered: pd.DataFrame):
    total_customers = len(df_full)
    filtered_customers = len(df_filtered)
    total_leads = int(df_full["lead_cnt"].fillna(0).sum())
    leads_in_view = int(df_filtered["lead_cnt"].fillna(0).sum())
    expected_conversion = df_filtered["ml_unified_customer_proba"].sum()
    avg_cltv = df_filtered["cltv_profit_final"].mean() if filtered_customers else 0

    cols = st.columns(4)
    metric_payload = [
        ("Total customers", f"{total_customers:,}", f"Leads {total_leads:,}"),
        ("Customers in view", f"{filtered_customers:,}", f"Leads {leads_in_view:,}"),
        ("Expected conversions", f"{expected_conversion:.1f}", "Sum(ML proba)"),
        ("Avg CLTV (₹)", f"{avg_cltv:,.0f}", "Filtered subset"),
    ]
    for col, payload in zip(cols, metric_payload):
        with col:
            st.markdown('<div class="crm-metric-card">', unsafe_allow_html=True)
            st.caption(payload[2])
            st.metric(payload[0], payload[1])
            st.markdown("</div>", unsafe_allow_html=True)


def build_product_category_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No data for selected filters.")
        return

    agg = (
        df.groupby("product_category")
        .agg(
            total_leads=("lead_cnt", "sum"),
            expected_converted=("ml_unified_customer_proba", "sum"),
        )
        .reset_index()
    )
    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("product_category", title="Product category"),
            y=alt.Y("total_leads", title="Total leads"),
            tooltip=["product_category", "total_leads", "expected_converted"],
            color=alt.Color("product_category", legend=None),
        )
    )
    st.altair_chart(chart, use_container_width=True)

    chart_conv = (
        alt.Chart(agg)
        .mark_bar(color="#0f9d58")
        .encode(
            x=alt.X("product_category", title="Product category"),
            y=alt.Y("expected_converted", title="Expected conversions"),
            tooltip=["product_category", "expected_converted", "total_leads"],
        )
    )
    st.altair_chart(chart_conv, use_container_width=True)


def build_product_drill(df: pd.DataFrame):
    if df.empty:
        st.info("No data for product drill.")
        return
    product = (
        df.groupby("product_name")
        .agg(
            total_leads=("lead_cnt", "sum"),
            avg_score=("hybrid_score_0_100_final", "mean"),
            cltv=("cltv_profit_final", "sum"),
        )
        .reset_index()
        .sort_values("total_leads", ascending=False)
    )
    st.dataframe(product, use_container_width=True, hide_index=True)


def build_pending_matrix(df: pd.DataFrame, dimensions: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=dimensions + LEAD_BUCKET_ORDER + ["Totals"])
    tmp = df.copy()
    grouped = (
        tmp.groupby(dimensions + ["lead_recency_bucket"])
        .size()
        .reset_index(name="pending_leads")
    )
    pivot = grouped.pivot_table(
        index=dimensions,
        columns="lead_recency_bucket",
        values="pending_leads",
        fill_value=0,
        aggfunc="sum",
    )
    pivot = pivot.reindex(columns=LEAD_BUCKET_ORDER, fill_value=0)
    pivot["Totals"] = pivot.sum(axis=1)
    return pivot.reset_index()


def render_customer_card(row: pd.Series, full_df: pd.DataFrame):
    st.markdown(f"### {row.get('full_name', 'Customer')} | `{row.get('cust_id')}`")
    info_cols = st.columns(3)
    info_cols[0].metric("Hybrid score", _fmt_int(row.get("hybrid_score_0_100_final")))
    info_cols[1].metric("ML probability", _fmt_float(row.get("ml_unified_customer_proba"), 3))
    info_cols[2].metric("CLTV (₹)", f"{_fmt_int(row.get('cltv_profit_final')):,}")

    profile_cols = st.columns(2)
    with profile_cols[0]:
        st.subheader("Customer profile", divider="gray")
        st.write(
            f"**Age / Gender**: {_fmt_int(row.get('age'))} / {row.get('gender', '—')}"
        )
        st.write(f"**Income segment**: {row.get('income_segment', '—')}")
        st.write(
            f"**Location**: {row.get('branch', '—')} · {row.get('center', '—')} · {row.get('zone', '—')}"
        )
        st.write(
            f"**Area type**: {row.get('urban_rural_flag', '—')} · Risk: {row.get('risk_bucket', '—')}"
        )
        st.write(f"**PIN / City**: {row.get('pin_code', '—')} / {row.get('region', '—')}")
        st.write(f"**Phone**: {row.get('phone_number', 'Not available')}")

    with profile_cols[1]:
        st.subheader("Relationship snapshot", divider="gray")
        st.write(f"**Products held**: {_fmt_int(row.get('num_products'))}")
        st.write(f"**Lead count**: {_fmt_int(row.get('lead_cnt'))}")
        st.write(f"**Last source**: {row.get('last_lead_source', '—')}")
        st.write(f"**Last lead date**: {row.get('last_lead_date', '—')}")
        st.write(f"**Recency bucket**: {row.get('lead_recency_bucket', '—')}")
        st.write(f"**Recommended action**: {row.get('status_label', row.get('action_bucket'))}")

        balances = [
            ("CBS balance", row.get("cbs_total_clr_bal_amt")),
            ("AA inflows", row.get("aa_avg_monthly_inflows")),
            ("AA outflows", row.get("aa_avg_monthly_outflows")),
        ]
        for label, value in balances:
            st.write(f"- {label}: {_fmt_float(value)}")

    st.write("---")
    st.dataframe(pd.DataFrame([row]), use_container_width=True)
    st.download_button(
        "Download customer (CSV)",
        full_df[full_df["cust_id"] == row["cust_id"]].to_csv(index=False).encode(),
        file_name=f"customer_{row['cust_id']}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
df_360_raw, df_summary, df_hero = load_dashboard_data()
df_360 = enrich_customer_frame(df_360_raw)

# Developer-facing snapshot so we know what data exists when iterating on UI.
with st.expander("Data snapshot · tables + key columns", expanded=False):
    st.markdown(
        """
**DuckDB sources**
- `df_360_raw` ⇢ `ans.customer_360_dashboard_v2`
- `df_summary` ⇢ `ans.customer_360_dashboard_summary`
- `df_hero` ⇢ `ans.customer_360_hero_slice`

**Shapes**
- df_360_raw: `{}` rows × `{}` cols
- df_summary: `{}` rows × `{}` cols
- df_hero: `{}` rows × `{}` cols

**Key customer fields (df_360_raw)**
`cust_id`, `full_name`, `gender`, `age`, `region`, `zone`, `priority_segment`,
`action_bucket`, `lead_cnt`, `last_lead_date`, `lead_recency_days`,
`lead_recency_bucket`, `last_lead_source`, `last_prod_product_id`,
`hybrid_score_0_100_final`, `ml_unified_customer_proba`, `cltv_profit_final`,
`cltv_decile_final`, `num_products`, `label_has_any_product`, `label_has_any_lead`

**Summary fields (df_summary)**
`priority_segment`, `action_bucket`, `lead_recency_bucket`,
`last_lead_source`, `n_customers`, `avg_hybrid_score`, `avg_ml_proba`, `avg_cltv`

**Hero slice fields (df_hero)**
Subset of df_360_raw with the same columns, pre-filtered for HIGH + recent digital.
        """.format(
            df_360_raw.shape[0],
            df_360_raw.shape[1],
            df_summary.shape[0],
            df_summary.shape[1],
            df_hero.shape[0],
            df_hero.shape[1],
        )
    )

latest_date = df_360["creation_date"].max()
date_label = latest_date.strftime("%d %b %Y") if pd.notna(latest_date) else "—"
st.title("CRM Leads Dashboard")
st.caption(f"Data as on {date_label}. DuckDB source · unified scoring + CLTV.")

filters = build_filter_controls(df_360)
df_view = apply_filters(df_360, filters)

render_metric_cards(df_360, df_view)

st.markdown("### All leads")
st.caption("Single customer = unified view of lead history, scoring and CLTV.")
table_cols = [
    ("cust_id", "Cust ID"),
    ("full_name", "Customer"),
    ("center", "Center"),
    ("zone", "Zone"),
    ("branch", "Branch"),
    ("product_name", "Product"),
    ("last_lead_source", "Source"),
    ("status_label", "Status"),
    ("lead_recency_label", "Lead age"),   # <-- use the friendly label
    ("hybrid_score_0_100_final", "Hybrid score"),
    ("ml_unified_customer_proba", "ML proba"),
    ("cltv_profit_final", "CLTV (₹)"),
]

existing_cols = [c for c, _ in table_cols if c in df_view.columns]
grid_df = df_view[existing_cols].rename(columns=dict(table_cols))
grid_df = grid_df.sort_values(
    ["Hybrid score", "ML proba", "CLTV (₹)"], ascending=[False, False, False]
).reset_index(drop=True)

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
gb.configure_grid_options(animateRows=True, rowHeight=32)
gb.configure_columns(
    {
        "Hybrid score": {
            "type": ["numericColumn"],
            "valueFormatter": "Math.round(value)",
            "width": 110,
            "maxWidth": 120,
        },
        "ML proba": {
            "type": ["numericColumn"],
            "valueFormatter": "Number(value).toFixed(3)",
            "width": 110,
            "maxWidth": 120,
        },
        "CLTV (₹)": {
            "type": ["numericColumn"],
            "valueFormatter": "Math.round(value)",
            "width": 110,
            "maxWidth": 120,
        },
    }
)

# Optional: also control some text columns so the grid looks tighter
gb.configure_column("Cust ID", width=90, pinned="left")
gb.configure_column("Customer", width=170)
gb.configure_column("Center", width=90)
gb.configure_column("Zone", width=80)
gb.configure_column("Branch", width=140)
gb.configure_column("Product", width=150)
gb.configure_column("Source", width=130)
gb.configure_column("Status", width=160)
gb.configure_column("Lead age", width=120)

grid_options = gb.build()

grid = AgGrid(
    grid_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    enable_enterprise_modules=False,
    height=420,
    theme="streamlit",
)

selected_rows = grid.get("selected_rows", [])
if hasattr(selected_rows, "to_dict"):
    selected_rows = selected_rows.to_dict(orient="records")
if isinstance(selected_rows, list) and selected_rows:
    selected_id = selected_rows[0].get("Cust ID")
    if selected_id and st.session_state.get("selected_customer") != selected_id:
        st.session_state["selected_customer"] = selected_id
        match = df_360[df_360["cust_id"] == selected_id]
        if not match.empty:

            @st.dialog(f"Customer {selected_id}")
            def _customer_modal():
                render_customer_card(match.iloc[0], df_360)

            _customer_modal()

st.download_button(
    "Download filtered leads (CSV)",
    df_view.to_csv(index=False).encode(),
    file_name="crm_leads_filtered.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("### Product category view")
c1, c2 = st.columns([1.2, 0.8])
with c1:
    build_product_category_chart(df_view)
with c2:
    build_product_drill(df_view)

st.markdown("---")
st.markdown("### Operational breakdowns")
tab_sources, tab_products, tab_summary = st.tabs(
    ["Source wise pending", "Product wise pending", "Segment summary"]
)

with tab_sources:
    source_table = build_pending_matrix(df_view, ["zone", "last_lead_source"])
    st.dataframe(source_table, use_container_width=True, hide_index=True)

with tab_products:
    product_table = build_pending_matrix(df_view, ["zone", "product_name"])
    st.dataframe(product_table, use_container_width=True, hide_index=True)

with tab_summary:
    st.caption("Priority × Action × Recency × Source (pre-aggregated view)")
    st.dataframe(df_summary, use_container_width=True)

st.markdown("---")
st.markdown("### Hero outreach lists")
st.caption("Pre-filtered HIGH priority + recent digital sources from DuckDB.")
st.dataframe(df_hero, use_container_width=True)
st.download_button(
    "Download hero outreach list (CSV)",
    df_hero.to_csv(index=False).encode(),
    file_name="customer_360_hero_slice_high_recent_digital.csv",
    mime="text/csv",
)
