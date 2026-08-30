"""
========================================================================
 PROBLEM STATEMENT 1 - SLEEP HEALTH ANALYTICS
 MODULE: Data Visualization (Person 3)
========================================================================

 PURPOSE:
    This script contains every reusable plotting function needed for
    the Sleep Health dashboard. It consumes the file produced by
    Person 2 ("processed_data.csv" = cleaned data + Sleep_Health_Tier)
    and returns Plotly figure objects that Person 4 drops straight
    into Streamlit with st.plotly_chart(fig, use_container_width=True).

 PIPELINE POSITION:
    Person 1 (Cleaning) --> Person 2 (Feature Engineering)
    --> [THIS SCRIPT: Person 3 - Visualization] --> Person 4 (Web App)
    --> Person 5 (Insights & Documentation)

 INPUT  : processed_data.csv  (output of Person 2's feature_engineering.py)
 OUTPUT : none on disk - every function returns a plotly Figure object
          for the caller (Person 4 / app.py) to render or export.

 IMPORTANT - Sleep_Health_Tier VALUES:
    Person 2's script (feature_engineering.py) writes these EXACT
    strings into the Sleep_Health_Tier column - every function below
    matches them precisely:
        "Tier 1 - Severely Deprived"
        "Tier 2 - Sub-Optimal"
        "Tier 3 - Healthy"
        "Unclassified"   (rows with leftover missing values; rare/edge case)

 NOTE: This script does NOT clean data or compute the tier column.
       It only reads processed_data.csv and visualizes it.
========================================================================
"""

# ------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------------
# 2. CONFIGURATION - column names & tier constants
# ------------------------------------------------------------------
# Edit these if your team's column headers differ from the standard
# Sleep Health and Lifestyle Dataset. Every function below references
# these constants only, so this is the single place to change things.
COL_OCCUPATION = "Occupation"
COL_SLEEP_DURATION = "Sleep Duration"
COL_QUALITY_OF_SLEEP = "Quality of Sleep"
COL_STRESS_LEVEL = "Stress Level"
COL_HEART_RATE = "Heart Rate"
COL_DAILY_STEPS = "Daily Steps"
COL_AGE = "Age"
COL_PHYSICAL_ACTIVITY = "Physical Activity Level"

TIER_COL = "Sleep_Health_Tier"

# Exact labels produced by Person 2's assign_sleep_tier(). Keep this
# order everywhere (charts, legends, sorting) so Tier 1 always reads
# as the most severe / most visually alarming category.
TIER_1 = "Tier 1 - Severely Deprived"
TIER_2 = "Tier 2 - Sub-Optimal"
TIER_3 = "Tier 3 - Healthy"
TIER_UNCLASSIFIED = "Unclassified"
TIER_ORDER = [TIER_1, TIER_2, TIER_3, TIER_UNCLASSIFIED]

# One consistent color palette used across every chart in the project.
TIER_COLORS = {
    TIER_1: "#E74C3C",            # red    - severely deprived
    TIER_2: "#F39C12",            # orange - sub-optimal
    TIER_3: "#27AE60",            # green  - healthy
    TIER_UNCLASSIFIED: "#95A5A6",  # grey   - missing / edge case
}

# Numeric columns used for the correlation heatmap.
CORRELATION_COLUMNS = [
    COL_AGE,
    COL_SLEEP_DURATION,
    COL_QUALITY_OF_SLEEP,
    COL_PHYSICAL_ACTIVITY,
    COL_STRESS_LEVEL,
    COL_HEART_RATE,
    COL_DAILY_STEPS,
]


# ------------------------------------------------------------------
# 3. LOAD DATASET (handed off from Person 2)
# ------------------------------------------------------------------
def load_processed_dataset(filepath: str = "processed_data.csv") -> pd.DataFrame:
    """
    Load Person 2's output file and order the Sleep_Health_Tier column
    as an ordered categorical so it always sorts/legends correctly.

    Parameters
    ----------
    filepath : str
        Path to processed_data.csv (cleaned data + Sleep_Health_Tier).

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[Visualization Error] Could not find '{filepath}'. "
            f"Make sure Person 2's processed_data.csv is in the same "
            f"folder as this script, or pass the correct path."
        )

    df = pd.read_csv(filepath)

    if TIER_COL not in df.columns:
        raise ValueError(
            f"[Visualization Error] '{filepath}' does not contain a "
            f"'{TIER_COL}' column. Make sure Person 2's feature "
            f"engineering step has run first."
        )

    present_categories = [t for t in TIER_ORDER if t in df[TIER_COL].unique()]
    df[TIER_COL] = pd.Categorical(df[TIER_COL], categories=present_categories, ordered=True)
    return df


# ------------------------------------------------------------------
# 4. SHARED STYLING HELPER
# ------------------------------------------------------------------
def _style(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    """
    Apply consistent, modern styling (white background, clean fonts,
    minimal gridlines, fixed height) to any Plotly figure produced by
    this module.

    IMPORTANT: every color/font set here is explicit on purpose. When
    Streamlit renders a figure with its default `theme="streamlit"`,
    it silently overrides layout colors (including font color) to
    match the app's active theme. If the app happens to be in dark
    mode, that override collides with the white background we set
    here and produces the washed-out, barely-readable text seen in
    the dashboard. The fix on the Streamlit side is to render with
    `theme=None` (done in app.py) so these settings are respected as-is.
    """
    full_title = title if not subtitle else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        title=dict(
            text=full_title,
            x=0.02,
            xanchor="left",
            font=dict(size=17, color="#1a2530"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial", size=13, color="#2c3e50"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11.5),
        ),
        margin=dict(l=50, r=30, t=90, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12),
        height=440,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d5dbdb", title_font=dict(size=13))
    fig.update_yaxes(showgrid=True, gridcolor="#ecf0f1", zeroline=False, title_font=dict(size=13))
    return fig


def _present_tiers(df: pd.DataFrame) -> list[str]:
    """Return TIER_ORDER filtered down to tiers actually present in df."""
    return [t for t in TIER_ORDER if t in set(df[TIER_COL].dropna().unique())]


# ------------------------------------------------------------------
# 5. REQUIRED VISUALIZATION 1 — Sleep Health Tier by Occupation
# ------------------------------------------------------------------
def prepare_occupation_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tier counts by occupation and order occupations by their
    Tier 1 (severely deprived) count, descending, so the most
    vulnerable professions appear first.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Long-format: Occupation, Sleep_Health_Tier, Count.
    """
    counts = (
        df.groupby([COL_OCCUPATION, TIER_COL], observed=True)
        .size()
        .reset_index(name="Count")
    )
    tier1_counts = (
        counts[counts[TIER_COL] == TIER_1]
        .set_index(COL_OCCUPATION)["Count"]
        .reindex(counts[COL_OCCUPATION].unique(), fill_value=0)
        .sort_values(ascending=False)
    )
    occupation_order = list(tier1_counts.index)
    counts[COL_OCCUPATION] = pd.Categorical(
        counts[COL_OCCUPATION], categories=occupation_order, ordered=True
    )
    return counts.sort_values(COL_OCCUPATION)


def plot_sleep_tier_by_occupation(df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart answering: "Which occupations experience the
    poorest sleep health?" Occupations are sorted so those with the
    most Tier 1 (severely deprived) individuals appear first.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Occupation and Sleep_Health_Tier columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data = prepare_occupation_data(df)
    fig = px.bar(
        data,
        x=COL_OCCUPATION,
        y="Count",
        color=TIER_COL,
        category_orders={TIER_COL: _present_tiers(df)},
        color_discrete_map=TIER_COLORS,
        hover_data={COL_OCCUPATION: True, TIER_COL: True, "Count": True},
    )
    fig.update_layout(
        barmode="stack",
        bargap=0.25,
        xaxis_title="Occupation",
        yaxis_title="Number of People",
    )
    fig.update_xaxes(tickangle=-30)
    return _style(
        fig,
        "Sleep Health Tier by Occupation",
        "Sorted by count of severely deprived (Tier 1) individuals",
    )


# ------------------------------------------------------------------
# 6. REQUIRED VISUALIZATION 2 — Daily Steps vs Sleep Duration
# ------------------------------------------------------------------
def plot_steps_vs_sleep(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot exploring: "Does physical activity relate to sleep
    duration?" Points are colored by Sleep_Health_Tier with an overall
    OLS trendline (requires the statsmodels package; falls back to a
    plain scatter if it isn't installed).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Daily Steps, Sleep Duration and Sleep_Health_Tier.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    hover_cols = [c for c in [COL_OCCUPATION, COL_STRESS_LEVEL] if c in df.columns]
    try:
        fig = px.scatter(
            df,
            x=COL_DAILY_STEPS,
            y=COL_SLEEP_DURATION,
            color=TIER_COL,
            category_orders={TIER_COL: _present_tiers(df)},
            color_discrete_map=TIER_COLORS,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="#34495e",
            hover_data=hover_cols,
        )
        # The OLS trendline is added as its own trace named "Overall
        # Trendline" by Plotly, which crowds the legend. Give it a
        # short, clearly-styled dashed line instead.
        for trace in fig.data:
            if "trendline" in (trace.name or "").lower() or trace.mode == "lines":
                trace.name = "Trend"
                trace.line.update(dash="dash", width=2)
                trace.showlegend = True
    except Exception:
        fig = px.scatter(
            df,
            x=COL_DAILY_STEPS,
            y=COL_SLEEP_DURATION,
            color=TIER_COL,
            category_orders={TIER_COL: _present_tiers(df)},
            color_discrete_map=TIER_COLORS,
            hover_data=hover_cols,
        )
    fig.update_traces(
        marker=dict(size=9, opacity=0.8, line=dict(width=1, color="white")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(xaxis_title="Daily Steps", yaxis_title="Sleep Duration (hours)")
    return _style(
        fig,
        "Daily Steps vs Sleep Duration",
        "Does physical activity relate to how much people sleep?",
    )


# ------------------------------------------------------------------
# 7. REQUIRED VISUALIZATION 3 — Stress Level vs Heart Rate
# ------------------------------------------------------------------
def plot_stress_vs_heart_rate(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot exploring: "Does higher stress correspond with higher
    heart rate?" Colored by Sleep_Health_Tier, with bubble size
    proportional to Daily Steps.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Stress Level, Heart Rate and Sleep_Health_Tier.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    hover_cols = [c for c in [COL_OCCUPATION, COL_DAILY_STEPS] if c in df.columns]
    fig = px.scatter(
        df,
        x=COL_STRESS_LEVEL,
        y=COL_HEART_RATE,
        color=TIER_COL,
        size=COL_DAILY_STEPS if COL_DAILY_STEPS in df.columns else None,
        size_max=22,  # caps bubble size so a few large-step outliers don't dominate/overlap the chart
        category_orders={TIER_COL: _present_tiers(df)},
        color_discrete_map=TIER_COLORS,
        hover_data=hover_cols,
    )
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="white")))
    fig.update_layout(xaxis_title="Stress Level", yaxis_title="Heart Rate (bpm)")
    return _style(
        fig,
        "Stress Level vs Heart Rate",
        "Bubble size = Daily Steps",
    )


# ------------------------------------------------------------------
# 8. BONUS VISUALIZATION 1 — Correlation Heatmap
# ------------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Annotated correlation heatmap across the key numeric lifestyle
    and health variables.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cols = [c for c in CORRELATION_COLUMNS if c in df.columns]
    corr = df[cols].corr().round(2)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_traces(textfont_size=11)
    fig.update_layout(coloraxis_colorbar=dict(title="Correlation", len=0.8))
    fig.update_xaxes(tickangle=-35)
    return _style(fig, "Correlation Heatmap", "Relationships between lifestyle & health metrics")


# ------------------------------------------------------------------
# 9. BONUS VISUALIZATION 2 — Sleep Duration Distribution
# ------------------------------------------------------------------
def plot_sleep_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Interactive histogram of Sleep Duration, split by Sleep_Health_Tier,
    with a marginal box plot for a quick view of spread/outliers.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = px.histogram(
        df,
        x=COL_SLEEP_DURATION,
        color=TIER_COL,
        category_orders={TIER_COL: _present_tiers(df)},
        color_discrete_map=TIER_COLORS,
        marginal="box",
        nbins=20,
        opacity=0.75,
    )
    fig.update_layout(
        barmode="overlay",
        bargap=0.05,
        xaxis_title="Sleep Duration (hours)",
        yaxis_title="Number of People",
    )
    return _style(fig, "Sleep Duration Distribution", "Split by Sleep Health Tier")


# ------------------------------------------------------------------
# 10. BONUS VISUALIZATION 3 — Sleep Duration by Occupation (Box Plot)
# ------------------------------------------------------------------
def plot_sleep_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Box plot showing the spread of Sleep Duration across occupations,
    ordered by median sleep duration (lowest first) to highlight the
    professions with the worst typical sleep.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    order = (
        df.groupby(COL_OCCUPATION)[COL_SLEEP_DURATION]
        .median()
        .sort_values()
        .index.tolist()
    )
    fig = px.box(
        df,
        x=COL_OCCUPATION,
        y=COL_SLEEP_DURATION,
        category_orders={COL_OCCUPATION: order},
        points="outliers",
    )
    fig.update_traces(
        marker=dict(color="#2980b9", size=5, opacity=0.7),
        line_color="#2c3e50",
        fillcolor="rgba(41, 128, 185, 0.25)",
    )
    fig.update_layout(xaxis_title="Occupation", yaxis_title="Sleep Duration (hours)")
    fig.update_xaxes(tickangle=-30)
    return _style(
        fig,
        "Sleep Duration by Occupation",
        "Occupations ordered by median sleep duration (lowest first)",
    )


# ------------------------------------------------------------------
# 11. QUICK STANDALONE PREVIEW (optional - not used by Streamlit)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Lets Person 3 sanity-check every chart locally without Streamlit.
    # Writes each chart to an HTML file instead of calling fig.show(),
    # so this works headlessly too.
    data = load_processed_dataset("processed_data.csv")

    charts = {
        "chart_1_tier_by_occupation.html": plot_sleep_tier_by_occupation(data),
        "chart_2_steps_vs_sleep.html": plot_steps_vs_sleep(data),
        "chart_3_stress_vs_heart_rate.html": plot_stress_vs_heart_rate(data),
        "chart_4_correlation_heatmap.html": plot_correlation_heatmap(data),
        "chart_5_sleep_distribution.html": plot_sleep_distribution(data),
        "chart_6_sleep_boxplot.html": plot_sleep_boxplot(data),
    }

    for filename, figure in charts.items():
        figure.write_html(filename)
        print(f"[OK] Saved preview: {filename}")

    print("\nAll charts generated. Open the .html files in a browser to preview.")