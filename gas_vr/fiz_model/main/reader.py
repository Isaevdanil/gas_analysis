"""Модуль для считывания и обработки все необходимых данных для расчета."""
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

import basic_funcs
from global_names import GlobalNames
from vfm_settings import VFM_Settings

gn = GlobalNames()


class Reader:
    """Класс для загрузки и хранения необходимых данных для расчета."""

    def __init__(self, data_load_type):
        """
        :param data_load_type: тип загружаемых данных
                               'local' - при загрузке локальных файлов
                               'DB' - загрузка с базы данных
        """
        self._data_load_type = data_load_type

        # определение типа загружаемых данных для инициализации соответствующего класса
        if "local" in self._data_load_type:
            self.settings = VFM_Settings()

            # инициализация переменных класса VFM_settings
            self.field = self.settings.field
            self.wells = self.settings.wells
            self.path_to_tr = self.settings.path_to_tr
            self.path_to_save_root = self.settings.full_path_to_save
            self.path_to_all_data = self.settings.full_path_to_all_data
            self.resample_time = self.settings.resample_time
            self.target = self.settings.target
            self.predict = self.settings.predict
            self.date_for_calc = pd.to_datetime(self.settings.date_for_calc)
            self.engine = self.settings.engine
            # определение номера скважины
            if type(self.wells) == str:
                self.well_name = self.wells
            # определение оставшихся атрибутов
            self._path_to_save = self.settings.todays_date_folder()
            self._file_name = self.file_name
            self._file_choke_d = self.file_choke_d
            self.data = self._get_dynamic_data()
            self.list_with_pvt_params = self._get_pvt_data()
            self.tr_data = basic_funcs.load_data(
                self.well_name, self.path_to_all_data, "tr"
            )
            self.inclinometry = basic_funcs.load_data(
                self.well_name, self.path_to_all_data, "inclinometry"
            )

            self.q_column_to_use = gn.q_gas_m3day
            self.q_data_type = VFM_Settings.check_q_gas_data(
                self.q_column_to_use, self.data
            )
            if self.settings.plot:
                self.create_graph()
        elif "DB" in self._data_load_type:
            pass

    def __repr__(self):
        return "Reader"

    def __str__(self):
        return "Data: " + str(dict(self.__dict__.items()))

    @property
    def file_choke_d(self):
        """
        Свойство получения имени файла с диаметрами штуцера

        :return: название файла
        """
        if self.field == "field_1":
            return "tr.pql"
        else:
            return None

    @file_choke_d.setter
    def file_choke_d(self, value: str):
        """
        Свойство задания имени файла, содержащего диаметры штуцера

        :param value: имя файла
        """
        self._file_choke_d = value

    @property
    def file_name(self):
        """
        Свойство получения текущего имени файла

        :return: имя файла
        """
        if self.field == "field_1":
            return "TM"

    @staticmethod
    def load_choke_data(
            path_to_all_data: str, well_name: str, file_choke_d: str
    ) -> pd.DataFrame:
        """
        Метод загрузки данных по диаметру штуцера

        :param path_to_all_data: путь к данным
        :param well_name: номер скважины
        :param file_choke_d: название файла

        :return: choke_data - датайрейм с диаметрами штуцера
        """
        tr_data = pd.read_pickle(
            path_to_all_data + well_name + "/" + file_choke_d, compression="gzip"
        )
        tr_data["CALC_DATE"] = pd.to_datetime(tr_data["CALC_DATE"])
        choke_data = pd.DataFrame(tr_data[["CALC_DATE", "CHOKE_DIAMETER"]])
        choke_data = choke_data.set_index("CALC_DATE")
        choke_data = choke_data.sort_index()

        return choke_data

    @staticmethod
    def past_choke_diam(
            data: pd.DataFrame, path_to_all_data: str, well_name: str, file_choke_d: str
    ) -> pd.DataFrame:
        """
        Метод добавления диаметра штуцера в основной датафрейм

        :param data: основной датафрейм
        :param path_to_all_data: путь к данным
        :param well_name: номер скважины
        :param file_choke_d: название файла

        :return: data - датайрейм с диаметрами штуцера
        """
        choke_data = Reader.load_choke_data(path_to_all_data, well_name, file_choke_d)
        data[gn.d_choke_mm] = None
        for i in range(choke_data.shape[0]):
            data.loc[data.index > choke_data.index[i], gn.d_choke_mm] = choke_data.iloc[
                i, 0
            ]
        data[gn.d_choke_mm] = data[gn.d_choke_mm].astype(int)

        return data

    def _get_dynamic_data(self) -> pd.DataFrame:
        """
        Метод получения датафрейма с динамическими данными

        :return: датафрейм с динамическими данными
        """
        data = basic_funcs.load_data(
            self.well_name, self.path_to_all_data, self.file_name
        )
        data = basic_funcs.rename_columns_by_dict(
            data, gn.return_dict_column_to_rename()
        )

        if self._file_choke_d != None:
            data = Reader.past_choke_diam(
                data, self.path_to_all_data, self.well_name, self._file_choke_d
            )

        data.index = pd.to_datetime(data.index)

        data = data[
            (data.index > self.date_for_calc[0]) & (data.index < self.date_for_calc[-1])
        ].copy()

        return data

    def _get_pvt_data(self) -> list:
        """
        Метод загрузки pvt данных

        :return: лист с pvt данными
        """
        if self.field == "field_1":
            return [0.91, 0.71, 1.023, 42.9, 58, 120, 1.1398, 2.77, None, None]

            df["gamma_oil_spl"] = 0.91
            df["gamma_gas_spl"] = 0.71
            df["gamma_wat_spl"] = 1.023
            df["rsb_m3m3_spl"] = 42.9
            df["tres_c_spl"] = 58
            df["pb_atma_spl"] = 120
            df["bob_m3m3_spl"] = 1.1398
            df["muob_cp_spl"] = 2.77
            df["rp_m3m3_sk"] = None

            df = pd.read_pickle(
                self.path_to_all_data + self.well_name + "/pvt.pql", compression="gzip"
            )
            return [
                float(df.loc["gamma_oil_spl", self.well_name]),
                float(df.loc["gamma_gas_spl", self.well_name]),
                float(df.loc["gamma_wat_spl", self.well_name]),
                float(df.loc["rsb_m3m3_spl", self.well_name]),
                float(df.loc["tres_c_spl", self.well_name]),
                float(df.loc["pb_atma_spl", self.well_name]),
                float(df.loc["bob_m3m3_spl", self.well_name]),
                float(df.loc["muob_cp_spl", self.well_name]),
                float(df.loc["rp_m3m3_sk", self.well_name]),
                None,
            ]

    def create_graph(self):
        """
        Метод для построения графика plotly
        :return: None
        """
        data = self.data

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
            title_text=f"Field={self.field}, Well={self.well_name}, Step=Reader",
            height=450 * 3,
        )

        fig.layout.hovermode = "x"

        plot(
            fig,
            filename=f"{self.settings.full_path_to_save}/{self.well_name}_init_data_view.html",
            auto_open=True,
        )

    @staticmethod
    def load_choke_data(
            path_to_all_data: str, well_name: str, file_choke_d: str
    ) -> pd.DataFrame:
        """
        Метод загрузки данных по диаметру штуцера

        :param path_to_all_data: путь к данным
        :param well_name: номер скважины
        :param file_choke_d: название файла

        :return: choke_data - датайрейм с диаметрами штуцера
        """
        tr_data = pd.read_pickle(
            path_to_all_data + well_name + "/" + file_choke_d, compression="gzip"
        )
        tr_data["CALC_DATE"] = pd.to_datetime(tr_data["CALC_DATE"])
        choke_data = pd.DataFrame(tr_data[["CALC_DATE", "CHOKE_DIAMETER"]])
        choke_data = choke_data.set_index("CALC_DATE")
        choke_data = choke_data.sort_index()

        return choke_data

    @staticmethod
    def past_choke_diam(
            data: pd.DataFrame, path_to_all_data: str, well_name: str, file_choke_d: str
    ) -> pd.DataFrame:
        """
        Метод добавления диаметра штуцера в основной датафрейм

        :param data: основной датафрейм
        :param path_to_all_data: путь к данным
        :param well_name: номер скважины
        :param file_choke_d: название файла

        :return: data - датайрейм с диаметрами штуцера
        """
        choke_data = Reader.load_choke_data(path_to_all_data, well_name, file_choke_d)
        data[gn.d_choke_mm] = None
        for i in range(choke_data.shape[0]):
            data.loc[data.index > choke_data.index[i], gn.d_choke_mm] = choke_data.iloc[
                i, 0
            ]
        data[gn.d_choke_mm] = data[gn.d_choke_mm].astype(int)

        return data
