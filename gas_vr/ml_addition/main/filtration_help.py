import datetime
import sys

import numpy as np
import pandas as pd

sys.path.append("../vfm_help_func")
from global_names import GlobalNames

gn = GlobalNames()


def check_columns(df: pd.DataFrame):
    """
    если замеров меньше 5 за весь промежуток, осредняет и заполняет весь ряд
    """
    print("Проверяем на количество замеров...")
    for column in df.columns:
        if len(df[column].unique()) < 5:
            print(column)
            df[column] = df[column].describe()["50%"]


def del_values_below_zero(df: pd.DataFrame):
    """
    Функция для удаления значений ниже и близких к 0
    """

    print("Удаляем значения, не входящие в границы...")
    for column in df.columns:
        if "Обводненность" in column or "обводненность" in column:
            df[column] = df[column].apply(
                lambda x: x if (x >= 0) | (x == np.nan) else np.nan
            )
        else:
            df[column] = df[column].apply(
                lambda x: x if (x > 0) | (x == np.nan) else np.nan
            )


def pressure_filtr(df: pd.DataFrame):
    """
    Проверка, что все значения давлений в атмосферах
    """

    print("Проверяем размерности...")
    for column in df.columns:
        if "давление" in column or "Давление" in column:
            print(f"Меняем значения для колонки {column}")
            df[column] = df[column].apply(lambda x: x if x >= 5 else x * 10)
        else:
            continue


def delete_emissions(df: pd.DataFrame, value: float, cols: list):
    """
    Удаление выбросов
    """

    print("Удаляем выбросы...")
    for column in cols:
        if len(df[column].unique()) <= 5:
            continue

        first_quartile = df[column].describe()["25%"]
        third_quartile = df[column].describe()["75%"]
        iqr = third_quartile - first_quartile

        df[column] = df[column].apply(
            lambda x: x
            if (x > (first_quartile - value * iqr))
               & (x < (third_quartile + value * iqr))
            else np.nan
        )


def data_split(df: pd.DataFrame, resample_time: str, column_where_dropna: str):
    """
    Разделение данных, 1 - с замерами дебита, 2 - без замеров
    """
    if resample_time != "-1":
        df = df.resample(resample_time).mean()
    df_2 = df.copy()
    df = df.dropna(subset=[column_where_dropna]).interpolate(
        method="linear", limit_direction="both"
    )
    df.dropna(axis="columns", how="all", inplace=True)

    list_to_del = []
    for one_index in df_2.index:
        if one_index in df.index:
            list_to_del.append(one_index)
    data_without_measures = df_2.drop(index=list_to_del)
    data_without_measures.dropna(axis="columns", how="all", inplace=True)

    return df, data_without_measures


def number_of_measures(df: pd.DataFrame, column_where_dropna: str):
    """
    Функция для получения данных о количестве интервалов замера и их времени
    """

    df_2 = df.copy()
    df_2 = df_2.dropna(subset=[column_where_dropna])

    list_index_start = [df_2.index[0]]
    list_index_end = []
    for i in range(df_2.shape[0] - 1):
        if (df_2.index[i + 1] - df_2.index[i]).total_seconds() > 4200:
            list_index_start.append(df_2.index[i + 1])
            list_index_end.append(df_2.index[i])
    list_index_end.append(df_2.index[-1])
    df_with_vv_starts_ends = pd.DataFrame(
        {"Начало": list_index_start, "Конец": list_index_end}
    )

    return df_with_vv_starts_ends


def get_df_list_with_unique_measure(
        df: pd.DataFrame,
        column_where_dropna: str,
        left_time_to_drop_min: int,
        right_time_to_drop_min: int,
):
    """
    Получение листа со всеми датафреймами по замерам
    """
    df_list = []
    df_with_vv_starts_ends = number_of_measures(df, column_where_dropna)

    left_delta = datetime.timedelta(minutes=left_time_to_drop_min)
    right_delta = datetime.timedelta(minutes=right_time_to_drop_min)

    for j in range(df_with_vv_starts_ends.shape[0]):
        this_df = df[
            (df.index > df_with_vv_starts_ends["Начало"].iloc[j])
            & (df.index < df_with_vv_starts_ends["Конец"].iloc[j])
            ]
        this_df = this_df[
            (this_df.index > (this_df.index[0] + left_delta))
            & (this_df.index < (this_df.index[-1] - right_delta))
            ]
        df_list.append(this_df)

    return df_list


def filtr_window(df, value, rolling_time, use_knn):
    this_df, roll_median, roll_std = get_roll_median_and_std(df, rolling_time)
    cols = this_df.columns

    for column in cols:
        upper_threshold = roll_median[column] + roll_std[column] * value
        lower_threshold = roll_median[column] - roll_std[column] * value
        this_df["drop_flag"] = np.where(
            (this_df[column] > upper_threshold) | (this_df[column] < lower_threshold),
            1,
            0,
        )
        this_df[column] = np.where(
            (this_df["drop_flag"] == 1), np.nan, this_df[column].values
        )
        this_df = this_df.drop(columns=["drop_flag"])
    if use_knn == False:
        this_df = this_df.interpolate(method="linear", limit_direction="both")

    return this_df


def filtr_window_2(df, value, rolling_time, create_roll_table=False):
    this_df = df.copy().sort_index()
    cols = df.columns
    for column in cols:
        this_df[column] = this_df[column].dropna()
        roll_median = this_df[column].rolling(window=rolling_time).median()
        roll_std = this_df[column].rolling(window=rolling_time).std()

        upper_threshold = roll_median + roll_std * value
        lower_threshold = roll_median - roll_std * value
        this_df["drop_flag"] = np.where(
            (this_df[column] > upper_threshold) | (this_df[column] < lower_threshold),
            1,
            0,
        )
        this_df[column] = np.where(
            (this_df["drop_flag"] == 1), np.nan, this_df[column].values
        )
        this_df = this_df.drop(columns=["drop_flag"])
    for column in gn.p_buf_atm, gn.p_lin_atm, gn.p_cas_atm:
        this_df[column] = this_df[column].interpolate(
            method="linear", limit_direction="both"
        )

    if create_roll_table == True:
        rolling_table = this_df.copy()
        rolling_table = rolling_table.rolling(window="8H").mean()
    else:
        rolling_table = None

    return this_df, rolling_table


def filtr_window_series(df, value, rolling_time, interpolation=False):
    this_df, roll_median, roll_std = get_roll_median_and_std(df, rolling_time)

    column = this_df.name
    this_df = this_df.to_frame(name=column)

    upper_threshold = roll_median + roll_std * value
    lower_threshold = roll_median - roll_std * value

    this_df["drop_flag"] = np.where(
        (this_df[column] > upper_threshold) | (this_df[column] < lower_threshold), 1, 0
    )
    this_df[column] = np.where(
        (this_df["drop_flag"] == 1), np.nan, this_df[column].values
    )
    this_df = this_df.drop(columns=["drop_flag"])
    if interpolation == True:
        this_df = this_df.interpolate(method="linear", limit_direction="both")

    return this_df


def get_roll_median_and_std(df, rolling_time):
    df = df.copy().dropna().sort_index()

    roll_median = df.rolling(window=rolling_time).median()
    roll_std = df.rolling(window=rolling_time).std()

    return df, roll_median, roll_std


def check_q_oil_data(df, column_type):
    if column_type == gn.q_liq_m3day:
        this_data = df.dropna(subset=[gn.q_liq_m3day])
        short_time, long_time = 0, 0
        for i in range(this_data.shape[0] - 1):
            if (this_data.index[i + 1] - this_data.index[i]).total_seconds() < 300:
                short_time += 1
            else:
                long_time += 1
        if long_time > short_time:
            print("Единичные замеры с АГЗУ")
            q_oil_data = "one"
        else:
            print("Дифференциальные замеры с АГЗУ")
            q_oil_data = "many"
    else:
        q_oil_data = "many"

    return q_oil_data


def clear_q_oil_data(df):
    this_data = df.dropna(subset=[gn.q_liq_m3day])
    list_to_del = []
    for i in range(this_data.shape[0] - 1):
        if (this_data.index[i + 1] - this_data.index[i]).total_seconds() < 7200:
            list_to_del.append(this_data[gn.q_liq_m3day].iloc[i + 1])
            print("Значение удалено")
    print(list_to_del)
    df[gn.q_liq_m3day] = df[gn.q_liq_m3day].where(~df[gn.q_liq_m3day].isin(list_to_del))

    return df


def delete_target_columns(df):
    col_list_to_del = [
        gn.q_liq_m3day,
        gn.q_oil_m3day,
        gn.q_oil_mass_tday,
        gn.q_gas_m3day,
        gn.gor_m3m3,
        gn.watercut_perc,
    ]
    for column in col_list_to_del:
        if column in df.columns:
            df = df.drop(columns=[column])

    return df


def add_dp(df):
    if gn.dp_atm not in df.columns:
        df[gn.dp_atm] = df[gn.p_buf_atm] - df[gn.p_lin_atm]


def add_wat_cut_and_gor(df):
    if gn.watercut_perc not in df.columns:
        df[gn.watercut_perc] = (1 - df[gn.q_oil_m3day] / df[gn.q_liq_m3day]) * 100
    if gn.gor_m3m3 not in df.columns:
        df[gn.gor_m3m3] = df[gn.q_gas_m3day] / df[gn.q_oil_m3day]
