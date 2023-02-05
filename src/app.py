# %% imports
# import json
import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import time
from helper import (
    generate_dates,
    convert_to_num,
    save_data,
    filter_raw,
    filter_col,
    pivot_table,
    rename_cols,
    fill_gaps,
    add_local_datetime,
)

# %% App Config
st.set_page_config(
    page_title="Elexon Data Visualise",
    page_icon="🎇",
    layout="centered",
    initial_sidebar_state="expanded",
)

# %% Configs: Define Constants and Configs
API_KEY = "lcwmesc8yu83703"
PERIOD = "*"
SERVICE_TYPE = "csv"
REPORTS = {
    "B1620": None,
}
FILTER_RAW = {}
FILTER_COL = {
    "B1620": [
        "SettlementDate",
        "SettlementPeriod",
        "year",
        "month",
        "day",
        "Quantity",
        "PowerSystemResourceType",
    ]
}
PIVOT = {
    "B1620": {
        "values": "Quantity",
        "index": [
            "SettlementDate",
            "SettlementPeriod",
            "year",
            "month",
            "day",
        ],
        "columns": "PowerSystemResourceType",
    },
}
RENAME = {
    # "B1620": {
    #     "Biomass": "Biomass",
    #     "Fossil Hard coal": "Fossil_Hard_Coal",
    #     "Fossil Gas": "Fossil_Gas",
    #     "Fossil Oil": "Fossil_Oil",
    #     "Hydro Pumped Storage": "Hydro_Pumped_Storage",
    #     "Hydro Run-of-river and poundage": "Hydro_River_Poundage",
    #     "Nuclear": "Nuclear",
    #     "Other": "Other",
    #     "Solar": "Solar",
    #     "Wind Onshore": "Wind_Onshore",
    #     "Wind Offshore": "Wind_Offshore",
    # }
}


# %% Functions


def call_api(url, params):
    """Call API and retries if call fails.
    Args:
        url (str): API URL
        params (dict): params for requests.get()
    Returns:
        res [requests.response]: Response from API
    """
    NO_OF_TRIES = 2
    for i in range(NO_OF_TRIES):
        res = requests.get(url, params=params)

        if res.ok and res.headers.get("Content-Disposition", "not").startswith(
            "attachment"
        ):
            return res
        else:
            time.sleep(0.5)
    return None


def fetch_data(
    report: str,
    cols: list = None,
    query_date: str = None,
) -> pd.DataFrame:
    """Fetch data from API and returns pandas DataFrame
    Args:
        report (str): Table name
        cols (list, optional): Columns names for tables which donot contain column names in API response. Defaults to None.
        query_date (str, optional): . Date filter for API. Defaults to None.
    Returns:
        pd.DataFrame: Converts API response to pandas DataFrame
    """
    params = {
        "APIKey": API_KEY,
        "Period": PERIOD,
        "SettlementPeriod": PERIOD,
        "SettlementDate": query_date,
        "FromDate": query_date,
        "ToDate": query_date,
        "FromSettlementDate": query_date,
        "ToSettlementDate": query_date,
        "ServiceType": SERVICE_TYPE,
    }

    url = f"https://api.bmreports.com/BMRS/{report}/v1"
    res = call_api(url, params=params)

    # Extract the data from Response (res)
    if res is not None:
        resp = res.content.decode("UTF-8")
        resp = [x.split(",") for x in resp.split("\n")]
        if cols is None:
            columns = resp[4]
            resp = resp[5:-1]
        else:
            columns = cols
            resp = resp[1:]

        # convert resp to datafarme
        data = pd.DataFrame(resp, columns=columns)

        # ---- clean / convert / sort ----
        # replace empty and NULL with np.nan
        data = (
            data.replace("", np.nan)
            .replace("NULL", np.nan)
            .replace('"', "", regex=True)
        )
        # remove spaces in column names
        data.columns = data.columns.str.replace(" ", "")
        # remove rows where "recordType" == "FTR"
        if "recordType" in data.columns:
            data = data[data["recordType"] != "FTR"]
        # convert Dates (str -> datetime)
        data["SettlementDate"] = pd.to_datetime(data["SettlementDate"])
        data["year"], data["month"], data["day"] = (
            data["SettlementDate"].dt.year,
            data["SettlementDate"].dt.month,
            data["SettlementDate"].dt.day,
        )
        # convert numbers (str -> int/float)
        data = convert_to_num(data)
        # sort by Date and Period
        if "SettlementPeriod" in data.columns:
            data.sort_values(
                by=["SettlementDate", "SettlementPeriod"],
                inplace=True,
                ignore_index=True,
            )
        else:
            data.sort_values(
                by=["SettlementDate"],
                inplace=True,
                ignore_index=True,
            )
        return data
    else:
        print(f"{report} on {query_date} not able to fetch")
        return pd.DataFrame()


def fetch_data_xml(
    report: str = "SYSWARN",
    from_date: str = None,
    to_date: str = None,
    service_type: str = "xml",
) -> pd.DataFrame:
    """Fetch table from ELEXON API in xml format and convert to pandas DataFrame
    Args:
        report (str, optional): Name of table. Defaults to 'SYSWARN'.
        from_date (str, optional): Start date. Defaults to '2019-01-01'.
        to_date (str, optional): End data. Defaults to today + 2 days.
        service_type (str, optional): Response type (csv or xml). Defaults to 'xml'.
    Returns:
        pd.DataFrame: XML data converted to pandas DataFrame
    """
    params = {
        "APIKey": API_KEY,
        "FromDate": from_date,
        "ToDate": to_date,
        "ServiceType": service_type,
    }

    url = f"https://api.bmreports.com/BMRS/{report}/v1"
    res = requests.get(url, params=params)

    if res.ok and res.headers.get("Content-Disposition", "not").startswith(
        "attachment"
    ):
        root = ET.fromstring(res.content.decode("UTF-8"))
        body = root[2][0]  # response body

        df_list = []
        for child in body:
            df_list.append({i.tag: i.text for i in child})

        data = pd.DataFrame(df_list)

        # ---- clean / convert / sort ----
        # replace empty and NULL with np.nan
        data = (
            data.replace("", np.nan)
            .replace("NULL", np.nan)
            .replace('"', "", regex=True)
        )
        # remove sapces in column names
        data.columns = data.columns.str.replace(" ", "")
        # remove rows where "recordType" == "FTR"
        if "recordType" in data.columns:
            data = data[data["recordType"] != "FTR"]
        # convert Dates (str -> datetime)
        data["warningDateTime"] = pd.to_datetime(data["warningDateTime"])
        data["year"], data["month"], data["day"] = (
            data["warningDateTime"].dt.year,
            data["warningDateTime"].dt.month,
            data["warningDateTime"].dt.day,
        )
        data = convert_to_num(data)
        return data
    else:
        print(f"{report} for {from_date} to {to_date} not able to fetch")
        return pd.DataFrame()


def fetch_tables(start_date, end_date) -> dict:
    """Fetch Data for all tables over the date range and stores them
    Args:
        start_date (str): Start date
        end_date (str): End date

    Returns:
        tables: a dictionary of table names and corresponding dataframe
    """
    tables = {}

    days = generate_dates(start_date, end_date)

    for rpt, cols in REPORTS.items():
        print(f"Fetching {rpt} for {start_date} to {end_date}.")
        if rpt in ["SYSWARN"]:
            df = fetch_data_xml(rpt, from_date=start_date, to_date=end_date)
        else:
            df = pd.DataFrame()
            for day in days:
                data = fetch_data(rpt, cols=cols, query_date=day)
                print(f"{rpt} on {day} has {data.shape[0]} lines")
                df = pd.concat([df, data], ignore_index=True)

        tables[rpt] = df

    return tables


# %% Side Bar
st.sidebar.header("User Settings")
st.sidebar.markdown("""---""")

# Get the date range
start_date = st.sidebar.date_input(
    "Start Date", dt.date.today() - dt.timedelta(days=2)
)
end_date = st.sidebar.date_input("End Date", dt.date.today())
if start_date > end_date:
    start_date, end_date = end_date, start_date

start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")


# Load and Clean Data
placeholder = st.sidebar.empty()
with placeholder.container():
    with st.spinner("Loading the data ..."):
        tables = fetch_tables(start_date, end_date)

placeholder.empty()
with placeholder.container():
    with st.spinner(f"Cleaning ELEXON data from {start_date} to {end_date}"):
        for rpt, df in tables.items():
            # st.write(f"Cleaning {rpt} over {start_date} to {end_date} ...")
            if df.empty:
                print(f"{rpt} over {start_date} to {end_date} has no records.")
                continue
            else:
                if rpt in ["SYSWARN"]:
                    pass
                else:
                    df = filter_raw(df, rpt, FILTER_RAW)
                    df = filter_col(df, rpt, FILTER_COL)
                    df = pivot_table(df, rpt, PIVOT)
                    df = rename_cols(df, rpt, RENAME)
                    df = fill_gaps(df)
                    df = add_local_datetime(df)
                    tables[rpt] = df


st.sidebar.write(f"Loaded {len(tables)} table(s)")


# %% Main Body
st.title("UK Energy Generation by Type")
st.markdown("___")

B1620 = tables["B1620"]
st.line_chart(
    B1620,
    x="local_datetime",
    y=[
        "Biomass",
        "Fossil Gas",
        "Fossil Hard coal",
        "Fossil Oil",
        "Hydro Pumped Storage",
        "Hydro Run-of-river and poundage",
        "Nuclear",
        "Other",
        "Solar",
        "Wind Offshore",
        "Wind Onshore",
    ],
    height=600,
    use_container_width=True,
)

# %%
