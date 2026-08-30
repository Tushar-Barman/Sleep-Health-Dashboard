"""
========================================================================
 PROBLEM STATEMENT 1 - SLEEP HEALTH ANALYTICS
 MODULE: Interactive Web Dashboard + Executive Summary (Person 4 & 5)
========================================================================

 PURPOSE:
    This is the final deliverable of the project. It is a single
    Streamlit application that:

      Person 4 (Web App Developer):
        - Loads processed_data.csv (Person 2's output)
        - Displays KPI cards (total records, tier distribution %,
          avg resting heart rate of Tier 1)
        - Lets the user filter the whole dashboard by Occupation and
          Gender
        - Renders every chart Person 3 built (visualization.py),
          reacting live to the active filters

      Person 5 (Summary Writer):
        - Programmatically (not hand-written) analyzes the filtered
          data every time the page runs
        - Surfaces 5-7 concise, data-driven insights
        - Surfaces 2-3 actionable lifestyle recommendations

 PIPELINE POSITION:
    Person 1 (Cleaning) -> Person 2 (Feature Engineering)
    -> Person 3 (Visualization) -> [THIS FILE: Person 4 + Person 5]

 INPUT  : processed_data.csv   (output of feature-engineer.py)
 RUN    : streamlit run app.py

 NOTE: This file does NOT re-clean data, does NOT recompute
       Sleep_Health_Tier, and does NOT redefine any chart already
       built in visualization.py. It only loads, filters, lays out,
       and interprets what Person 1-3 already produced.
========================================================================
"""

# ------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

# Every chart used on this dashboard is Person 3's — we only import
# and call these, we never redefine plotting logic here.
from visualization import (
    COL_OCCUPATION,
    COL_SLEEP_DURATION,
    COL_QUALITY_OF_SLEEP,
    COL_STRESS_LEVEL,
    COL_HEART_RATE,
    COL_DAILY_STEPS,
    TIER_COL,
    TIER_1,
    TIER_2,
    TIER_3,
    load_processed_dataset,
    plot_sleep_tier_by_occupation,
    plot_steps_vs_sleep,
    plot_stress_vs_heart_rate,
    plot_correlation_heatmap,
    plot_sleep_distribution,
    plot_sleep_boxplot,
)

PROCESSED_FILE = "processed_data.csv"
COL_GENDER = "Gender"
COL_SLEEP_DISORDER = "Sleep Disorder"


# ------------------------------------------------------------------
# 2. PAGE CONFIG (must be the first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Sleep Health Analytics Dashboard",
    page_icon="😴",
    layout="wide",
)


# ------------------------------------------------------------------
# 3. DATA LOADING (cached so the CSV is only read once per session)
# ------------------------------------------------------------------
@st.cache_data
def get_data(filepath: str) -> pd.DataFrame:
    """
    Loads Person 2's processed_data.csv via Person 3's own loader
    (load_processed_dataset), so we inherit the exact same
    Sleep_Health_Tier category ordering used by every chart.

    ADDED FIX (flagging this — not in the original spec):
    Plain pd.read_csv() treats the literal text "None" as a missing
    value by default. Person 1's cleaning script writes "None" into
    Sleep Disorder for people with no recorded disorder, but reloading
    the CSV anywhere downstream (including Person 3's own loader)
    silently turns those back into NaN. Person 1's script even prints
    a warning about this exact gotcha. Since the executive summary
    below reports Sleep Disorder prevalence, this dashboard restores
    those values to the string "None" immediately after loading so
    the KPIs/insights are accurate. No other column or value is
    touched.
    """
    df = load_processed_dataset(filepath)
    if COL_SLEEP_DISORDER in df.columns:
        df[COL_SLEEP_DISORDER] = df[COL_SLEEP_DISORDER].fillna("None")
    return df


try:
    raw_df = get_data(PROCESSED_FILE)
except FileNotFoundError:
    st.error(
        f"Could not find '{PROCESSED_FILE}'. Run Person 1's cleaning "
        f"script, then Person 2's feature-engineer.py, before launching "
        f"this dashboard."
    )
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# ------------------------------------------------------------------
# 4. SIDEBAR FILTERS (dynamically narrow every KPI / chart / insight)
# ------------------------------------------------------------------
st.sidebar.header("🔎 Filters")

occupation_options = sorted(raw_df[COL_OCCUPATION].dropna().unique().tolist())
selected_occupations = st.sidebar.multiselect(
    "Occupation", options=occupation_options, default=occupation_options
)

gender_options = sorted(raw_df[COL_GENDER].dropna().unique().tolist()) if COL_GENDER in raw_df.columns else []
selected_genders = st.sidebar.multiselect(
    "Gender", options=gender_options, default=gender_options
) if gender_options else []

st.sidebar.caption(
    "Filters apply to the KPIs, every chart below, and the "
    "executive summary at the bottom of the page."
)

df = raw_df.copy()
if selected_occupations:
    df = df[df[COL_OCCUPATION].isin(selected_occupations)]
if selected_genders:
    df = df[df[COL_GENDER].isin(selected_genders)]

if df.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters to see data.")
    st.stop()


# ------------------------------------------------------------------
# 5. HEADER
# ------------------------------------------------------------------
st.title("😴 Sleep Health Analytics Dashboard")
st.markdown(
    "Translating sleep, vitals, and lifestyle metrics into actionable "
    "recovery categories — built on top of the cleaned dataset, the "
    "`Sleep_Health_Tier` classification, and the chart library produced "
    "earlier in the pipeline."
)


# ------------------------------------------------------------------
# 6. KPI CARDS
# ------------------------------------------------------------------
st.header("📊 Overview")

total_records = len(df)
tier_pct = (df[TIER_COL].value_counts(normalize=True) * 100).round(1)
tier1_df = df[df[TIER_COL] == TIER_1]
avg_hr_tier1 = tier1_df[COL_HEART_RATE].mean() if not tier1_df.empty else np.nan

kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Records", f"{total_records:,}")
kpi_cols[1].metric("Tier 1 — Severely Deprived", f"{tier_pct.get(TIER_1, 0.0):.1f}%")
kpi_cols[2].metric("Tier 2 — Sub-Optimal", f"{tier_pct.get(TIER_2, 0.0):.1f}%")
kpi_cols[3].metric("Tier 3 — Healthy", f"{tier_pct.get(TIER_3, 0.0):.1f}%")
kpi_cols[4].metric(
    "Avg Resting HR (Tier 1)",
    f"{avg_hr_tier1:.0f} bpm" if pd.notna(avg_hr_tier1) else "N/A",
)


# ------------------------------------------------------------------
# 7. VISUAL INSIGHTS (every chart is Person 3's — we only call them)
# ------------------------------------------------------------------
st.header("📈 Visual Insights")

st.subheader("Sleep Tiers Across Professions")
st.plotly_chart(plot_sleep_tier_by_occupation(df), width='stretch')

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Daily Steps vs Sleep Duration")
    st.plotly_chart(plot_steps_vs_sleep(df), width='stretch')
with col_b:
    st.subheader("Stress Level vs Heart Rate")
    st.plotly_chart(plot_stress_vs_heart_rate(df), width='stretch')

with st.expander("📌 More charts (correlation, distribution, spread by occupation)"):
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(plot_correlation_heatmap(df), width='stretch')
        st.plotly_chart(plot_sleep_boxplot(df), width='stretch')
    with col_d:
        st.plotly_chart(plot_sleep_distribution(df), width='stretch')


# ------------------------------------------------------------------
# 8. EXECUTIVE SUMMARY (Person 5 — computed programmatically)
# ------------------------------------------------------------------
def build_executive_summary(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Analyzes the (filtered) dataset and returns:
        insights        -> 5-7 short, data-driven observations
        recommendations -> 2-3 actionable lifestyle recommendations

    Every number quoted here is computed live from `data` — nothing
    is hard-coded, so the summary always matches the active filters.
    """
    insights: list[str] = []

    # --- Insight 1: most vulnerable occupation (highest Tier 1 share) ---
    occ_tier1_rate = (
        data.groupby(COL_OCCUPATION, observed=True)[TIER_COL]
        .apply(lambda s: (s == TIER_1).mean() * 100)
        .sort_values(ascending=False)
    )
    if not occ_tier1_rate.empty and occ_tier1_rate.iloc[0] > 0:
        top_occ = occ_tier1_rate.index[0]
        insights.append(
            f"**{top_occ}** has the highest share of severely sleep-deprived "
            f"individuals at **{occ_tier1_rate.iloc[0]:.0f}%** of that group "
            f"falling into Tier 1."
        )

    # --- Insight 2: occupation with the best sleep health ---
    occ_tier3_rate = (
        data.groupby(COL_OCCUPATION, observed=True)[TIER_COL]
        .apply(lambda s: (s == TIER_3).mean() * 100)
        .sort_values(ascending=False)
    )
    if not occ_tier3_rate.empty:
        best_occ = occ_tier3_rate.index[0]
        insights.append(
            f"**{best_occ}** shows the healthiest sleep profile overall, with "
            f"**{occ_tier3_rate.iloc[0]:.0f}%** of that group rated Tier 3 (Healthy)."
        )

    # --- Insight 3: stress vs sleep duration relationship ---
    if data[COL_STRESS_LEVEL].nunique() > 1 and data[COL_SLEEP_DURATION].nunique() > 1:
        stress_sleep_corr = data[COL_STRESS_LEVEL].corr(data[COL_SLEEP_DURATION])
        direction = "inversely" if stress_sleep_corr < 0 else "positively"
        insights.append(
            f"Stress Level and Sleep Duration are **{direction} correlated** "
            f"(r = {stress_sleep_corr:.2f}) — "
            + ("higher stress tends to come with shorter sleep." if stress_sleep_corr < 0
               else "the relationship is weaker/opposite of the typical pattern.")
        )

    # --- Insight 4: activity vs sleep quality relationship ---
    activity_col = "Physical Activity Level"
    if activity_col in data.columns and data[activity_col].nunique() > 1:
        activity_quality_corr = data[activity_col].corr(data[COL_QUALITY_OF_SLEEP])
        insights.append(
            f"Physical Activity Level and Quality of Sleep have a correlation of "
            f"**r = {activity_quality_corr:.2f}**, suggesting "
            + ("more active individuals tend to report better sleep quality."
               if activity_quality_corr > 0.1 else
               "activity level alone is a weak predictor of sleep quality in this data.")
        )

    # --- Insight 5: heart rate gap between Tier 1 and Tier 3 ---
    hr_by_tier = data.groupby(TIER_COL, observed=True)[COL_HEART_RATE].mean()
    if TIER_1 in hr_by_tier.index and TIER_3 in hr_by_tier.index:
        hr_gap = hr_by_tier[TIER_1] - hr_by_tier[TIER_3]
        insights.append(
            f"Tier 1 individuals average **{hr_by_tier[TIER_1]:.0f} bpm** resting "
            f"heart rate versus **{hr_by_tier[TIER_3]:.0f} bpm** for Tier 3 — a gap "
            f"of **{hr_gap:.0f} bpm**, consistent with poor sleep straining "
            f"cardiovascular recovery."
        )

    # --- Insight 6: daily steps gap between tiers ---
    steps_by_tier = data.groupby(TIER_COL, observed=True)[COL_DAILY_STEPS].mean()
    if TIER_1 in steps_by_tier.index and TIER_3 in steps_by_tier.index:
        steps_gap = steps_by_tier[TIER_3] - steps_by_tier[TIER_1]
        insights.append(
            f"Tier 3 (Healthy) individuals average **{steps_by_tier[TIER_3]:,.0f} "
            f"daily steps** compared to **{steps_by_tier[TIER_1]:,.0f}** for Tier 1 "
            f"— a difference of **{steps_gap:,.0f} steps/day**."
        )

    # --- Insight 7: sleep disorder prevalence in Tier 1 ---
    if COL_SLEEP_DISORDER in data.columns and not tier1_df.empty:
        t1_disorder_rate = (data.loc[data[TIER_COL] == TIER_1, COL_SLEEP_DISORDER] != "None").mean() * 100
        overall_disorder_rate = (data[COL_SLEEP_DISORDER] != "None").mean() * 100
        insights.append(
            f"**{t1_disorder_rate:.0f}%** of Tier 1 individuals have a recorded "
            f"sleep disorder, versus **{overall_disorder_rate:.0f}%** across the "
            f"full filtered population."
        )

    # ---------------- Recommendations (derived from the above) ----------------
    recommendations: list[str] = []

    if not occ_tier1_rate.empty and occ_tier1_rate.iloc[0] > 0:
        recommendations.append(
            f"Prioritize workplace wellness programs (flexible hours, wind-down "
            f"breaks, stress management resources) for **{occ_tier1_rate.index[0]}**, "
            f"the occupation with the highest Tier 1 rate."
        )

    if activity_col in data.columns and data[activity_col].corr(data[COL_QUALITY_OF_SLEEP]) > 0.1:
        recommendations.append(
            "Encourage consistent daily physical activity — even modest increases "
            "in step count correlate with better sleep duration and quality in "
            "this population."
        )

    if data[COL_STRESS_LEVEL].corr(data[COL_SLEEP_DURATION]) < 0:
        recommendations.append(
            "Introduce stress-reduction initiatives (mindfulness sessions, "
            "workload review, counseling access) as a lever for improving sleep "
            "duration, given the inverse relationship observed between stress "
            "and sleep."
        )

    if not recommendations:
        recommendations.append(
            "Continue monitoring sleep, stress, and activity metrics across "
            "occupations to catch emerging risk patterns early."
        )

    return insights, recommendations


st.header("📝 Executive Summary")
st.caption("Generated live from the currently filtered dataset — not a static write-up.")

insight_list, recommendation_list = build_executive_summary(df)

st.subheader("Key Insights")
for point in insight_list:
    st.markdown(f"- {point}")

st.subheader("Actionable Recommendations")
for rec in recommendation_list:
    st.markdown(f"- {rec}")

st.divider()
st.caption(
    "Data pipeline: Person 1 (cleaning) → Person 2 (Sleep_Health_Tier) → "
    "Person 3 (charts) → Person 4/5 (this dashboard + summary)."
)
