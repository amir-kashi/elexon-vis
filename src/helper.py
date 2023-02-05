"""Generic helper functions for AWS glue"""

# %% Imports
"""Import Required Libraries"""
import datetime as dt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow
from functools import reduce

filesystem = None

# %% Functions


def generate_dates(start_date: str, end_date: str):
    """Generates list of dates
    Args:
        start_date (str): Start date
        end_date (str): End date

    Returns:
        List: List of dates
    """
    sdate = dt.datetime.strptime(start_date, "%Y-%m-%d")
    edate = dt.datetime.strptime(end_date, "%Y-%m-%d") + dt.timedelta(days=1)

    return [
        (sdate + dt.timedelta(days=x)).strftime("%Y-%m-%d")
        for x in range((edate - sdate).days)
    ]


def convert_to_num(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Numeric Values to INT or FLOAT"""
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            continue
        elif col in ["SettlementPeriod", "year", "month", "day"]:
            df[col] = df[col].astype(int)
        else:
            try:
                df[col] = df[col].astype(np.float64)
            except:
                pass
        continue

    return df


def save_data(
    df: pd.DataFrame,
    rpt: str,
    path: str,
    partition_cols: list = None,
    pq_filename: str = None,
):
    """Write pandas DataFrame in parquet format to S3 bucket

    Args:
        df (pd.DataFrame): DataFrame to save
        rpt (str): Table name
        path (str): local or AWS S3 path to store the parquet files
        partition_cols (list, optional): Columns used to partition the parquet files. Defaults to None.
        pq_filename (str): name of the .parquet files
    """
    if pq_filename == None:
        pq_filename = rpt

    if partition_cols is None:
        pq.write_to_dataset(
            pyarrow.Table.from_pandas(df),
            path + f"/{rpt}",
            filesystem=filesystem,
        )
    else:
        pq.write_to_dataset(
            pyarrow.Table.from_pandas(df),
            path + f"/{rpt}",
            filesystem=filesystem,
            partition_cols=partition_cols,
            partition_filename_cb=lambda x: f"{pq_filename}.parquet",
        )


def read_parquet_tables(
    rpt: str,
    start_date: str,
    end_date: str,
    path: str,
    pq_filename: str = None,
) -> pd.DataFrame:
    """Read parquet file partitions

    Args:
        rpt (str): Table name
        start_date (str): starting date to clean the data (%Y-%m-%d)
        end_date (str): ending date to clean the data (%Y-%m-%d)
        path (str): local or AWS S3 path to read the parquet files
        pq_filename (str): name of the .parquet files
    Returns:
        pd.DataFrame: Datframe from parquet file
    """
    if pq_filename == None:
        pq_filename = rpt

    # convert date range to list of dates
    date_list = generate_dates(start_date, end_date)

    df = pd.DataFrame()
    for read_date in date_list:
        # convert date to integers for filters
        r_year = dt.datetime.strptime(read_date, "%Y-%m-%d").year
        r_month = dt.datetime.strptime(read_date, "%Y-%m-%d").month
        r_day = dt.datetime.strptime(read_date, "%Y-%m-%d").day

        try:
            data = (
                pq.ParquetDataset(
                    path
                    + f"/{rpt}/year={r_year}/month={r_month}/day={r_day}/{pq_filename}.parquet",
                    filesystem=filesystem,
                )
                .read_pandas()
                .to_pandas()
            )
            data["year"], data["month"], data["day"] = r_year, r_month, r_day
        except:
            continue

        df = pd.concat([df, data], ignore_index=True)

    # for AWS Glue
    if "__index_level_0__" in df.columns:
        df.drop("__index_level_0__", axis=1, inplace=True)

    return df


def filter_raw(df: pd.DataFrame, rpt: str, FILTER_RAW: dict) -> pd.DataFrame:
    """filter raws bases on pre-defined rules"""
    if rpt in FILTER_RAW.keys():
        for col, values in FILTER_RAW[rpt].items():
            df = df[df[col].apply(lambda x: True if x in values else False)]
    return df


def filter_col(df: pd.DataFrame, rpt: str, FILTER_COL: dict) -> pd.DataFrame:
    """filter columns"""
    if rpt in FILTER_COL.keys():
        df = df[FILTER_COL[rpt]]
    return df


def pivot_table(df: pd.DataFrame, rpt: str, PIVOT: dict) -> pd.DataFrame:
    if rpt in PIVOT.keys():
        group_args = PIVOT[rpt]
        df = (
            pd.pivot_table(
                df,
                values=group_args["values"],
                index=group_args["index"],
                columns=group_args["columns"],
                aggfunc="last",
            )
            .rename_axis(None, axis=1)
            .reset_index()
        )
    return df


def rename_cols(df: pd.DataFrame, rpt: str, RENAME: dict) -> pd.DataFrame:
    if rpt in RENAME.keys():
        df.rename(columns=RENAME[rpt], inplace=True)
    return df


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Find missed data in the dataframe and fill the gaps with interpolated data"""
    # Check if the df has 'SettlementPeriod' (special case for daily data acquisition)
    if "SettlementPeriod" not in df.columns:
        period_list = np.reshape(
            [list(range(1, 49)) for _ in range(len(df))], -1
        )
        df = pd.DataFrame(df.values.repeat(48, axis=0), columns=df.columns)
        df["SettlementPeriod"] = period_list
    # drop duplicates in the df
    df.drop_duplicates(
        ["SettlementDate", "SettlementPeriod"], keep="last", inplace=True
    )
    # drop first day(s) if data are incomplete
    while df.iloc[0]["SettlementPeriod"] != 1:
        df = df[df["SettlementDate"] > df.iloc[0]["SettlementDate"]]

    # extract info about date range
    first_day = df["SettlementDate"].min()
    last_day = df["SettlementDate"].max()
    n_days = (last_day - first_day).days
    last_day_periods = min(
        df[df.SettlementDate == last_day]["SettlementPeriod"].max(), 48
    )

    # create date and period list for the base_df
    date_list = []
    period_list = []
    for i in range(n_days):
        date_list.extend(np.repeat(first_day + dt.timedelta(days=i), 48))
        period_list.extend(list(range(1, 49)))

    date_list.extend(np.repeat(last_day, last_day_periods))
    period_list.extend(list(range(1, last_day_periods + 1)))

    # initialise base_df
    base_df = pd.DataFrame(
        {"SettlementDate": date_list, "SettlementPeriod": period_list}
    )

    # merge df with base_df
    df_filled = pd.merge(
        base_df,
        df,
        how="left",
        on=["SettlementDate", "SettlementPeriod"],
    )

    # fill gaps in year, month, day
    df_filled["year"], df_filled["month"], df_filled["day"] = (
        df_filled["SettlementDate"].dt.year,
        df_filled["SettlementDate"].dt.month,
        df_filled["SettlementDate"].dt.day,
    )

    # list of numeric columns
    numeric_columns = df_filled.select_dtypes(include=np.number).columns

    # interpolate
    df_filled.loc[:, numeric_columns] = df_filled.loc[
        :, numeric_columns
    ].interpolate(method="linear", limit_area="inside")

    return df_filled


def add_local_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'local_datetime' to the tables based on
    SettlementDate and SettlementPeriod
    """
    # store original columns
    cols = df.columns
    # SettlementPeriod -> minutes
    df["temp_time"] = df["SettlementPeriod"].map(
        lambda x: str(dt.timedelta(minutes=(x - 1) * 30))
    )
    # concat SettlementDate and minutes
    df["local_datetime"] = pd.to_datetime(
        df["SettlementDate"].astype(str) + " " + df["temp_time"]
    )
    # remove temporary column
    df.drop("temp_time", axis=1, inplace=True)
    # sort columns (put 'local_datetime' in the beginning)
    cols = df.columns.to_list()
    cols.remove("local_datetime")
    sorted_cols = ["local_datetime"]
    sorted_cols.extend(cols)
    df = df[sorted_cols]

    return df


def merge_tables(tables: dict, rpt_base: str = None) -> pd.DataFrame:
    """Merge all dataframes in the tables
    Args:
        tables (dict): a dictionary of report names (key), and dataframes (values)
        rpt_base (str): base report for the merge
    Return:
        pd.Dataframe: merged dataframes
    """
    if len(tables) == 1:
        # returns the only dataframe available in the tables
        return list(tables.values())[0]

    if (rpt_base is None) or (rpt_base not in tables.keys()):
        # find the dataframe with the newest data
        rpt_base = list(tables.keys())[0]
        for rpt in tables.keys():
            if (
                tables[rpt]["local_datetime"].max()
                > tables[rpt_base]["local_datetime"].max()
            ):
                rpt_base = rpt

    # create the list of the reports (ordered)
    other_reports = list(tables.keys())
    other_reports.remove(rpt_base)
    reports_ordered = [rpt_base]
    reports_ordered.extend(other_reports)

    # create the list of dataframes (ordered)
    dfs = [tables[rpt] for rpt in reports_ordered]

    # merge all dataframes to one
    merged = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=[
                "local_datetime",
                "SettlementDate",
                "SettlementPeriod",
                "year",
                "month",
                "day",
            ],
            how="left",
        ),
        dfs,
    )

    return merged


## FPT HELPER FUNCTIONS


def read_single_parquet(
    rpt: str,
    path: str,
):
    """
        read data parquet in s3
    Args:
        rpt: resource save data
        path: path s3

    Returns:

    """
    data = (
        pq.ParquetDataset(
            path + f"/{rpt}",
            filesystem=filesystem,
        )
        .read_pandas()
        .to_pandas()
    )
    return data


def save_met_data(
    df: pd.DataFrame,
    rpt: str,
    path: str,
    date_str: dt.datetime,
    partition_cols: list = None,
):
    """Write pandas DataFrame in parquet format to S3 bucket

    Args:
        df (pd.DataFrame): DataFrame to save
        rpt (str): Table name
        date_str: datetime when save data
        path (str): local or AWS S3 path to store the parquet files
        partition_cols (list, optional): Columns used to partition the parquet files. Defaults to None.
    """
    year = date_str.year
    month = date_str.month
    date = date_str.day
    if partition_cols is None:
        pq.write_to_dataset(
            pyarrow.Table.from_pandas(df),
            path + f"/{rpt}",
            filesystem=filesystem,
            partition_filename_cb=lambda x: f"{rpt}_{year}_{month}_{date}.parquet",
        )
    else:
        pq.write_to_dataset(
            pyarrow.Table.from_pandas(df),
            path + f"/{rpt}",
            filesystem=filesystem,
            partition_cols=partition_cols,
            partition_filename_cb=lambda x: f"{rpt}.parquet",
        )
