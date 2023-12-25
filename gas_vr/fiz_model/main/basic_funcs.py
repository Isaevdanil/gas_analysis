"""
Модуль с базовыми функциями для ВР штуцер
"""
import datetime
import os

import pandas as pd

from global_names import GlobalNames

gn = GlobalNames()


def create_folders(path: str, well_list: list):
    """
    функция
    """
    try:
        for well in well_list:
            this_path = path + well
            os.mkdir(this_path)
    except:
        pass


def rename_columns_by_dict(df: pd.DataFrame, columns_name_dict: dict):
    """
    функция для переименования названий колонок, используя общий словарь из class_global_names
    """
    for i in df.columns:
        for items in columns_name_dict.items():
            if i in items[1] or i in [x.replace(" ", "") for x in items[1]]:
                print(f"Переименование колонки {i}")
                df = df.rename(columns={i: items[0]})
    return df


def get_result_input_df(well_name: str, df: pd.DataFrame):
    """
    функция для выделения данных для единичной скважины
    """
    if well_name == "-1":
        print("Выбраны все скважины для расчета")
        this_df = df.copy()

    else:
        this_df = df[df[gn.well_name] == well_name]

    this_df = this_df.dropna(axis="columns", how="all").set_index(gn.time)
    this_df.drop(columns=[gn.well_name], inplace=True)
    this_df = this_df.sort_index()

    return this_df


def load_data(well_name: str, path: str, file_name: str):
    """Метод для чтения динамических данных."""
    path = path + well_name + "/" + file_name + ".pql"
    df = pd.read_pickle(path, compression="gzip")
    return df


def save_data(well_name: str, path: str, file_name: str, df: pd.DataFrame):
    """
    функция для сохранения данных в файл pickle
    -1 используется в случае, если нужно сохранить таблицу с данными по всем скважинам
    """
    if well_name == "-1":
        path = path + file_name + ".pql"
    else:
        path = path + well_name + "/" + file_name + ".pql"
    df.to_pickle(path, compression="gzip", protocol=3)


def cut_df_with_getting_date(data, bounds):
    if bounds == "-1":
        return data
    else:
        left_bound = datetime.datetime.strptime(bounds[0], "%Y-%m-%d")
        right_bound = datetime.datetime.strptime(bounds[1], "%Y-%m-%d")
        print(left_bound, right_bound)
        return data[(data.index >= left_bound) & (data.index <= right_bound)]
