# %% imports
# import json
import datetime as dt
import pandas as pd
import streamlit as st
from helper import read_parquet_tables

# %% App Config
st.set_page_config(
    page_title="SCGC AI Power Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# %% Configs: Define Constants and Configs
COLS_TIME = ["local_datetime", "SettlementPeriod", "year", "month", "day"]
COLS_INTRADAY = [
    "niv_predicted_1sp",
    "niv_predicted_2sp",
    "niv_predicted_3sp",
    "niv_predicted_4sp",
    "niv_predicted_5sp",
    "niv_predicted_6sp",
    "niv_predicted_7sp",
    "niv_predicted_8sp",
]
COL_MORNING = ["dayahead_morning"]
COL_AFTERNOON = ["dayahead_afternoon"]
NIV = "ImbalanceQuantity(MAW)(B1780)"


# %% Functions


def prediction_strength(x):
    lower, middle, upper = THRESHOLDS

    if x > upper:
        return 1.5
    elif x > middle:
        return 1.0
    elif x > lower:
        return 0.5
    elif x > -lower:
        return 0.0
    elif x > -middle:
        return -0.5
    elif x > -upper:
        return -1.0
    elif x <= -upper:
        return -1.5
    else:
        return None


def cal_pnl(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Profit and Loss

    Args:
        predictions (pd.DataFrame): dataframe with the intraday and
                                    day ahead predictions
        Returns:
            pd.DataFrame: corresponding profit and loss values
    """
    df_pnl = predictions.copy()
    # Calcualte PnL for intraday predictions
    for col in COLS_INTRADAY:
        df_pnl[col] = df_pnl[col].apply(prediction_strength) * (
            df_pnl["ImbalancePriceAmount(B1770)"]
            - df_pnl["marketIndexPrice(MID)"]
        )

    # Calculate PnL for Day Ahead Predictions
    for col in COL_MORNING + COL_AFTERNOON:
        df_pnl[col] = df_pnl[col].apply(prediction_strength) * (
            df_pnl["ImbalancePriceAmount(B1770)"] - df_pnl["price_act(DAA)"]
        )

    # rename columns
    df_pnl = df_pnl[COLS_TIME + COLS_INTRADAY + COL_MORNING + COL_AFTERNOON]
    df_pnl = df_pnl.rename(
        {
            "niv_predicted_1sp": "pnl_1sp",
            "niv_predicted_2sp": "pnl_2sp",
            "niv_predicted_3sp": "pnl_3sp",
            "niv_predicted_4sp": "pnl_4sp",
            "niv_predicted_5sp": "pnl_5sp",
            "niv_predicted_6sp": "pnl_6sp",
            "niv_predicted_7sp": "pnl_7sp",
            "niv_predicted_8sp": "pnl_8sp",
            "dayahead_morning": "pnl_morning",
            "dayahead_afternoon": "pnl_afternoon",
        },
        axis="columns",
    )

    return df_pnl


# %% Side Bar
st.sidebar.header("User Settings")
st.sidebar.markdown("""---""")

# Get the date range
start_date = st.sidebar.date_input(
    "Start Date", dt.date.today() - dt.timedelta(days=2)
)
end_date = st.sidebar.date_input(
    "End Date", dt.date.today() + dt.timedelta(days=1)
)
if start_date > end_date:
    start_date, end_date = end_date, start_date

start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")


# %% Main Body
st.title("SCGC AI Power Trader")

tab_market, tab_niv, tab_pnl = st.tabs(
    ["Market Overview", "NIV Predicrions", "Profit and Loss"]
)

with tab_market:
    st.markdown("## UK Energy Generation by Type")
    # Load data
    B1620 = read_parquet_tables(
        rpt="B1620",
        start_date=start_date,
        end_date=end_date,
        path="data/cleaned/elexon",
        bucket_name="scgc",
    )
    # Line Chart
    st.line_chart(
        B1620,
        x="local_datetime",
        y=[
            "Fossil_Gas",
            "Nuclear",
            "Solar",
            "Wind_Offshore",
            "Wind_Onshore",
            "Biomass",
            "Fossil_Hard_Coal",
            "Fossil_Oil",
            "Hydro_Pumped_Storage",
            "Hydro_River_Poundage",
            "Other",
        ],
        height=600,
        use_container_width=True,
    )

with tab_niv:
    st.markdown("## NIV Predictions")
    # Load data
    lgbm_regression = read_parquet_tables(
        rpt="lgbm_regression",
        start_date=start_date,
        end_date=end_date,
        path="data/predictions",
        bucket_name="scgc",
    )
    # Select prediction
    preds_to_plot = st.multiselect(
        label="Select from available predictions",
        options=COL_MORNING + COL_AFTERNOON + COLS_INTRADAY,
        default=COL_MORNING,
        max_selections=4,
    )
    # Line Chart
    st.line_chart(
        lgbm_regression.set_index("local_datetime")[start_date:end_date],
        y=["ImbalanceQuantity(MAW)(B1780)"] + preds_to_plot,
        height=600,
        use_container_width=True,
    )

with tab_pnl:
    st.markdown("## Profit and Loss")

    col_pnl_input, col_pnl_chart = st.columns([1, 3], gap="large")

    with col_pnl_input:
        # Select the time range
        start_date_pnl = st.date_input(
            "Start Date Range", dt.date.today() - dt.timedelta(days=30)
        )
        end_date_pnl = st.date_input(
            "End Date Range", dt.date.today() + dt.timedelta(days=1)
        )
        if start_date_pnl > end_date_pnl:
            start_date_pnl, end_date_pnl = end_date_pnl, start_date_pnl

        start_date_pnl = start_date_pnl.strftime("%Y-%m-%d")
        end_date_pnl = end_date_pnl.strftime("%Y-%m-%d")

        # Select Thresholds
        st.markdown("___")
        st.markdown(
            """
            ### Select Thresholds
            Select the lower, middle, and upper levels for predicted NIV values
            """
        )
        lower = st.slider("Lower threshold", 0, 400, 25)
        middle = st.slider("Middle threshold", 0, 400, 150)
        upper = st.slider("Upper threshold", 0, 400, 250)
        THRESHOLDS = (lower, middle, upper)

    with col_pnl_chart:
        # Load Data
        lgbm_regression = read_parquet_tables(
            rpt="lgbm_regression",
            start_date=start_date_pnl,
            end_date=end_date_pnl,
            path="data/predictions",
            bucket_name="scgc",
        )
        # Calculate and visualise the PnL
        df_pnl = cal_pnl(lgbm_regression)
        df_pnl.fillna(0.0, inplace=True)

        # Select PnL's
        pnl_to_plot = st.multiselect(
            label="Select the PnL's to plot",
            options=[col for col in df_pnl.columns if col.startswith("pnl")],
            default=["pnl_8sp", "pnl_morning", "pnl_afternoon"],
            max_selections=4,
        )

        # Plot the PnL chart
        st.line_chart(
            df_pnl.set_index("local_datetime")[
                start_date_pnl:end_date_pnl
            ].cumsum(),
            y=["pnl_8sp", "pnl_morning", "pnl_afternoon"],
            height=600,
            use_container_width=True,
        )
