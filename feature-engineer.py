"""
========================================================================
 PROBLEM STATEMENT 1 - SLEEP HEALTH ANALYTICS
 MODULE: Feature Engineering (Person 2)
========================================================================

 PURPOSE:
    This script takes the CLEANED dataset produced by Person 1
    ("cleaned_data.csv") and adds a new column, "Sleep_Health_Tier",
    which classifies every individual into one of three sleep health
    tiers based on Sleep Duration, Quality of Sleep, and Stress Level.

 PIPELINE POSITION:
    Person 1 (Cleaning) --> [THIS SCRIPT: Person 2 - Feature Engineering]
    --> Person 3 (Visualization) --> Person 4 (Web App) --> Person 5 (Summary)

 INPUT  : cleaned_data.csv
 OUTPUT : processed_data.csv  (cleaned_data.csv + Sleep_Health_Tier column)

 NOTE: This script does NOT re-clean or modify existing data values.
       It only reads the cleaned file and appends the new tier column.
========================================================================
"""

# ------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import os
import sys


# ------------------------------------------------------------------
# 2. CONFIGURATION (file names kept as constants for easy editing)
# ------------------------------------------------------------------
INPUT_FILE = "cleaned_data.csv"
OUTPUT_FILE = "processed_data.csv"

# Columns that MUST be present in the cleaned dataset for this
# module to work. If any of these are missing, we cannot proceed.
REQUIRED_COLUMNS = ["Sleep Duration", "Quality of Sleep", "Stress Level"]


# ------------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------------
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the cleaned dataset produced by Person 1.
    Raises a clear, readable error if the file cannot be found or read.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[Feature Engineering Error] Could not find '{filepath}'. "
            f"Make sure Person 1's cleaned dataset is in the same folder "
            f"as this script, or update the INPUT_FILE path."
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(
            f"[Feature Engineering Error] Failed to read '{filepath}': {e}"
        )

    if df.empty:
        raise ValueError(
            f"[Feature Engineering Error] '{filepath}' was loaded but "
            f"contains no rows."
        )

    return df


# ------------------------------------------------------------------
# 4. VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------------
def validate_columns(df: pd.DataFrame, required_cols: list) -> None:
    """
    Ensures all columns needed for tier classification exist in the
    dataframe. Raises a clear error listing exactly what is missing,
    so Person 1 / the team can fix the upstream data if needed.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"[Feature Engineering Error] Missing required column(s): "
            f"{missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    print("[OK] All required columns are present:", required_cols)


# ------------------------------------------------------------------
# 5. TIER CLASSIFICATION LOGIC
# ------------------------------------------------------------------
def assign_sleep_tier(row: pd.Series) -> str:
    """
    Classifies a single row (individual) into a Sleep_Health_Tier.

    Priority order is STRICT and must not be changed:
        1. Tier 1 (Severely Deprived) checked FIRST
        2. Tier 2 (Sub-Optimal) checked SECOND (only if not Tier 1)
        3. Tier 3 (Healthy) is the default fallback

    Rules (exactly as per problem statement):
        Tier 1: Sleep Duration < 6.0
                OR (Sleep Duration < 6.5 AND Quality of Sleep <= 5)

        Tier 2: Sleep Duration < 7.0 AND Stress Level >= 6
                (and NOT already Tier 1)

        Tier 3: Everyone else
    """
    sleep_duration = row["Sleep Duration"]
    quality_of_sleep = row["Quality of Sleep"]
    stress_level = row["Stress Level"]

    # --- Guard: handle any unexpected missing values gracefully ---
    # (Person 1 should have cleaned these, but we defend against
    #  leftover NaNs so the script never crashes mid-run.)
    if pd.isna(sleep_duration) or pd.isna(quality_of_sleep) or pd.isna(stress_level):
        return "Unclassified"

    # --- TIER 1: Severely Deprived (checked first / highest priority) ---
    is_tier_1 = (sleep_duration < 6.0) or (
        sleep_duration < 6.5 and quality_of_sleep <= 5
    )
    if is_tier_1:
        return "Tier 1 - Severely Deprived"

    # --- TIER 2: Sub-Optimal (only reached if NOT Tier 1) ---
    is_tier_2 = (sleep_duration < 7.0) and (stress_level >= 6)
    if is_tier_2:
        return "Tier 2 - Sub-Optimal"

    # --- TIER 3: Healthy / Rested (default fallback) ---
    return "Tier 3 - Healthy"


# ------------------------------------------------------------------
# 6. APPLY CLASSIFICATION TO THE WHOLE DATASET
# ------------------------------------------------------------------
def apply_tier_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies assign_sleep_tier() row-by-row and adds the result as a
    new column: Sleep_Health_Tier.
    """
    df = df.copy()  # avoid mutating the caller's dataframe in place
    df["Sleep_Health_Tier"] = df.apply(assign_sleep_tier, axis=1)
    return df


# ------------------------------------------------------------------
# 7. VALIDATION / SANITY CHECKS
# ------------------------------------------------------------------
def print_validation_report(df: pd.DataFrame) -> None:
    """
    Prints a distribution of tiers and a few sample rows so the team
    can visually confirm the classification looks correct before it
    is handed off to Person 3 (Visualization).
    """
    print("\n" + "=" * 60)
    print("SLEEP HEALTH TIER — DISTRIBUTION")
    print("=" * 60)

    tier_counts = df["Sleep_Health_Tier"].value_counts()

    for tier in tier_counts.index:
        print(f"{tier:<30} : {tier_counts[tier]:>5} records")

    if "Unclassified" in tier_counts.index:
        print(
            "\n[WARNING] Some rows had missing Sleep Duration / "
            "Quality of Sleep / Stress Level values and were marked "
            "'Unclassified'. Flag this to Person 1."
        )

    print("\n" + "=" * 60)
    print("SAMPLE ROWS (first 10)")
    print("=" * 60)
    sample_cols = ["Sleep Duration", "Quality of Sleep", "Stress Level",
                    "Sleep_Health_Tier"]
    print(df[sample_cols].head(10).to_string(index=False))
    print("=" * 60 + "\n")


# ------------------------------------------------------------------
# 8. SAVE OUTPUT
# ------------------------------------------------------------------
def save_dataset(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the processed dataframe (cleaned data + Sleep_Health_Tier)
    to disk for the next stage of the pipeline (Person 3).
    """
    df.to_csv(filepath, index=False)
    print(f"[OK] Processed dataset saved to '{filepath}' "
          f"({len(df)} rows, {len(df.columns)} columns).")


# ------------------------------------------------------------------
# 9. MAIN PIPELINE
# ------------------------------------------------------------------
def main():
    print("Starting Feature Engineering module (Person 2)...\n")

    # Step 1: Load cleaned data from Person 1
    df = load_dataset(INPUT_FILE)
    print(f"[OK] Loaded '{INPUT_FILE}' with {len(df)} rows, "
          f"{len(df.columns)} columns.")

    # Step 2: Validate required columns exist
    validate_columns(df, REQUIRED_COLUMNS)

    # Step 3: Apply Sleep_Health_Tier classification
    df = apply_tier_classification(df)
    print("[OK] Sleep_Health_Tier column created.")

    # Step 4: Print validation / distribution report
    print_validation_report(df)

    # Step 5: Save processed output for Person 3
    save_dataset(df, OUTPUT_FILE)

    print("\nFeature Engineering complete. Handoff-ready for Visualization (Person 3).")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"\n[FAILED] {err}", file=sys.stderr)
        sys.exit(1)
