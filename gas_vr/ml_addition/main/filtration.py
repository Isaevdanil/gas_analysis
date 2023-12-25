"""
Модуль для фильтрации динамических данных
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from global_names import GlobalNames
from vfm_settings import VFM_Settings

gn = GlobalNames()


class Filtration:
    """Класс для фильтрации динамических данных."""

    def __init__(
            self,
            data: pd.DataFrame,
            settings: VFM_Settings,
            q_column_to_use: str,
            resample_time: str,
            field: str,
            well: str,
            value_for_check_columns: int = 5,
            value_for_delete_emissions: int = 3,
            left_time_to_drop_min: int = 60,
            right_time_to_drop_min: int = 60,
    ):
        """
        :param data: Данные
        :param settings: Класс с настройками
        :param q_column_to_use: Параметр для восстановления
        :param resample_time: Время осреднения
        :param field: Месторождение
        :param value_for_check_columns: определяет минимальное количество данных в столбцах, ниже которого метод
                                        произойдет заполнение колонки медианным значением
        :param value_for_delete_emissions: определяет коэффициент при сигма для удаления выбросов,
                                           рекомендованное значение в диапазоне [1.5, 3]
        :param left_time_to_drop_min: время в минутах для обрезания замера дебита слева
        :param right_time_to_drop_min: время в минутах для обрезания замера дебита справа
        """
        # self.reader = Reader(data_load_type=data_load_type)
        # инициализация переменных класса Reader
        self.data = data
        self.settings = settings
        self.filterable_data = None
        self.q_column = q_column_to_use
        self.resample_time = resample_time
        self.field = field
        self.well_name = well
        # Создания атрибутов для хранения данных с замерами и без
        self.df_with_liq_rates = None
        self.df_without_liq_rates = None
        self.list_of_dfs_with_liq_rates = None

        # Инициализация настроек фильтрации
        self.value_for_check_columns = value_for_check_columns
        self.value_for_delete_emissions = value_for_delete_emissions

        self.left_time_to_drop_min = left_time_to_drop_min
        self.right_time_to_drop_min = right_time_to_drop_min

        self.run_filtration()

        if self.settings.plot:
            self.create_graph()

    def create_graph(self):
        """
        Метод построения графика
        """

        data = self.filterable_data

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Дебит газа, м3/сут",
                "Давления, атм",
                "Диаметр штуцера, мм",
            ),
            shared_xaxes=True,
            vertical_spacing=0.01,
        )

        fig.update_xaxes(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            )
        )

        for this_col in [gn.q_gas_m3day]:
            if this_col in data.columns:
                this_df = data[this_col].dropna()
                fig.add_trace(
                    go.Scattergl(
                        x=this_df.index,
                        y=this_df,
                        name=this_col,
                        mode="lines+markers",
                        hovertemplate=this_col + ": %{y:.3f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        for this_col in [gn.p_buf_atm, gn.p_lin_atm]:
            if this_col in data.columns:
                this_df = data[this_col].dropna()
                fig.add_trace(
                    go.Scattergl(
                        x=this_df.index,
                        y=this_df,
                        name=this_col,
                        mode="lines+markers",
                        hovertemplate=this_col + ": %{y:.3f}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

        for this_col in [gn.d_choke_mm]:
            if this_col in data.columns:
                this_df = data[this_col].dropna()
                fig.add_trace(
                    go.Scattergl(
                        x=this_df.index,
                        y=this_df,
                        name=this_col,
                        mode="lines+markers",
                        hovertemplate=this_col + ": %{y:.3f}<extra></extra>",
                    ),
                    row=3,
                    col=1,
                )

        fig.update_layout(
            title_text=f"Field={self.field}, Well={self.well_name},Step=Filter",
            height=450 * 3,
        )

        fig.layout.hovermode = "x"

        plot(
            fig,
            filename=f"{self.settings.full_path_to_save}/{self.well_name}_after_filtration_view.html",
            auto_open=True,
        )

    @staticmethod
    def check_columns(df: pd.DataFrame, value=5):
        """
        Метод осреднения колонок, если слишком мало значений.

        Parameters
        ----------
        :param df: датафрейм с динамическими данными
        :param value: значение, ниже которого колонка будет осредняться,
                      по умолчанию value = 5
        """
        for column in df.columns:
            if len(df[column].unique()) < value:
                print(column)
                df[column] = df[column].describe()["50%"]

    def del_values_below_zero(self):
        """
        Метод удаление значений ниже 0
        """
        for column in self.filterable_data.columns:
            self.filterable_data[column] = self.filterable_data[column].apply(
                lambda x: x if (x > 0) | (x == np.nan) else np.nan
            )

            if column == gn.d_choke_mm:
                self.filterable_data[column] = (
                    self.filterable_data[column]
                    .ffill()
                    .bfill()
                )

    def pressure_filtr(self):
        """
        Метод проверки размерностей
        """
        for column in self.filterable_data.columns:
            if column in [gn.p_buf_atm, gn.p_lin_atm]:
                print(f"Меняем значения для колонки {column}")
                self.filterable_data[column] = self.filterable_data[column].apply(
                    lambda x: x if x >= 5 else x * 10
                )
            else:
                continue

    def delete_emissions(self):
        """
        Метод удаления выбросов по всему диапазону
        """

        value = self.value_for_delete_emissions

        for column in self.filterable_data.columns:
            if len(self.filterable_data[column].unique()) <= 5:
                continue

            first_quartile = self.filterable_data[column].describe()["25%"]
            third_quartile = self.filterable_data[column].describe()["75%"]
            iqr = third_quartile - first_quartile

            self.filterable_data[column] = self.filterable_data[column].apply(
                lambda x: x
                if (x > (first_quartile - value * iqr))
                   & (x < (third_quartile + value * iqr))
                else np.nan
            )

    def data_split(self):
        """
        Метод разделения данных: 1 - данные с замерами таргета;
                                 2 - данные без замеров

        Parameters
        ----------
        :param df: датафрейм с динамическими данными
        :param resample_time: время осреднения данных
        :param column_where_dropna: колонка с таргетом (по которой будет разделение)

        :return: 1) датафрейм, где происходили замеры таргета;
                 2) датафрейм, где замеров не было
        """
        resample_time = self.resample_time
        column_where_dropna = self.q_column
        df = self.filterable_data

        if resample_time != "-1":
            df = df.resample(resample_time).mean()
        self.df_without_liq_rates = df.copy()
        number_all = self.df_without_liq_rates.shape[0]
        print("Количество замеров:", number_all)
        self.df_with_liq_rates = df.dropna(subset=[column_where_dropna]).interpolate(
            method="linear", limit_direction="both"
        )
        number_q = self.df_with_liq_rates.shape[0]
        print("Количество замеров Qг:", number_q)

        if number_q < 1:
            raise Exception("Недостаточно замеров Qг - стоп")

    @staticmethod
    def number_of_measures(df: pd.DataFrame, column_where_dropna: str) -> pd.DataFrame:
        """
        Метод получения данных о количестве интервалов замера и их времени

        Parameters
        ----------
        :param df: датафрейм с данными
        :param column_where_dropna: колонка с таргетом

        :return: датафрейм с началом и концом всех замеров
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

    def add_dp(self):
        """
        Метод добавления в датафрейм колонки с перепадом давления
        """
        if gn.dp_atm not in self.filterable_data.columns:
            self.filterable_data[gn.dp_atm] = (
                    self.filterable_data[gn.p_buf_atm] - self.filterable_data[gn.p_lin_atm]
            )

    def add_wat_cut_and_gor(self):
        """
        Метод добавления в датафрейм газового фактора и обводненности

        :param df: датафрейм с данными
        """
        if gn.watercut_perc not in self.filterable_data.columns:
            self.filterable_data[gn.watercut_perc] = (1) * 100
        if gn.gor_m3m3 not in self.filterable_data.columns:
            self.filterable_data[gn.gor_m3m3] = 100

    def run_filtration(self):
        """
        Метод расчета полного цикла фильтрации
        """
        self.filterable_data = self.data
        self.del_values_below_zero()
        self.pressure_filtr()
        self.delete_emissions()
        self.add_dp()
        self.add_wat_cut_and_gor()
        self.data_split()
