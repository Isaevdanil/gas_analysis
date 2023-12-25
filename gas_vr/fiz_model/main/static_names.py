import sys

import pandas as pd

sys.path.append("../vfm_help_func")
from global_names import GlobalNames


class StaticData:
    def __init__(self):
        """
        Класс-структура для хранения данных о скважине из техрежима
        :param row: строка, извлеченная с техрежима для данной скважины
        """
        self._list_with_pvt_params = None
        self.company_name = None
        self.subcompany_name = None
        self.well_name = None
        self.well_work_mode = None

        self.well_type = None

        self.tr_filename = None
        self.tr_dataframe = None

        self.field_name_str = None
        self.reservoir_name_str = None

        self.d_cas_mm = None
        self.d_tube_mm = None
        self.d_choke_mm = None

        self.gamma_oil = None
        self.gamma_gas = None
        self.gamma_wat = None
        self.rsb_m3m3 = None
        self.tres_c = None
        self.pb_atm = None
        self.bob_m3m3 = None
        self.muob_cp = None
        self.rp_m3m3 = None
        self.watercut_perc = None

        self.qliq_m3day = None

        # средние значения
        self.watercut_perc_median = None
        self.gor_median = None
        self.q_gas_median = None
        self.t_lin_median = None

    @property
    def list_with_pvt_params(self):
        return self._list_with_pvt_params

    @list_with_pvt_params.setter
    def list_with_pvt_params(self, list_):
        self._list_with_pvt_params = list_

    def load_static_data(self, static_excel_full_path):
        # print('\n' + f"Загрузка статичных данных из  {static_excel_full_path} \n")
        static_df = pd.read_excel(static_excel_full_path, index_col=0)

        self.company_name = static_df[0].company_name
        self.subcompany_name = static_df[0].subcompany_name
        self.well_name = static_df[0].well_name
        self.well_work_mode = static_df[0].well_work_mode
        self.well_type = static_df[0].well_type

        self.tr_filename = static_df[0].tr_filename
        self.tr_dataframe = None
        self.field_name_str = static_df[0].field_name_str
        self.reservoir_name_str = static_df[0].reservoir_name_str
        self.d_cas_mm = static_df[0].d_cas_mm
        self.d_tube_mm = static_df[0].d_tube_mm
        self.d_choke_mm = static_df[0].d_choke_mm

        self.gamma_oil = static_df[0].gamma_oil
        self.gamma_gas = static_df[0].gamma_gas
        self.gamma_wat = static_df[0].gamma_wat
        self.rsb_m3m3 = static_df[0].rsb_m3m3
        self.tres_c = static_df[0].tres_c
        self.pb_atm = static_df[0].pb_atm
        self.bob_m3m3 = static_df[0].bob_m3m3
        self.muob_cp = static_df[0].muob_cp
        self.rp_m3m3 = static_df[0].rp_m3m3
        self.watercut_perc = static_df[0].watercut_perc
        self.qliq_m3day = static_df[0].qliq_m3day

    def get_median_values_from_TM_data(self, df, wat_cut_type):
        gn = GlobalNames()

        if wat_cut_type == "hal":
            self.watercut_perc_median = df[gn.watercut_perc_hal].median()
        elif wat_cut_type == "approved":
            self.watercut_perc_median = df[gn.watercut_perc_approved].median()
        else:
            self.watercut_perc_median = df[gn.watercut_perc].median()

        try:
            self.gor_median = df[gn.gor_m3m3].median()
        except:
            df[gn.gor_m3m3] = df[gn.q_gas_m3day] / (
                    df[gn.q_oil_mass_tday] / self.gamma_oil
            )
            self.gor_median = df[gn.gor_m3m3].median()
        self.q_gas_median = df[gn.q_gas_m3day].median()

        try:
            self.t_lin_median = df[gn.t_fluid_c].median()
        except:
            self.t_lin_median = 20

    def get_median_values_from_VX_data(self, df):
        gn = GlobalNames()

        try:
            self.watercut_perc_median = df[gn.watercut_perc_VX].median()
        except:
            self.watercut_perc_median = 0
        try:
            self.gor_median = df[gn.gor_m3m3_VX].median()
        except:
            self.gor_median = df[gn.gor_m3t_VX].median() / self.gamma_oil

        self.q_gas_median = df[gn.q_gas_m3day_VX].median()
        self.t_lin_median = df[gn.t_lin_C_VX].median()

    def __fill_by_true_tr(self, row):
        print("\n" + f"Прочитаем данные из техрежима для скважины {self.well_name}")
        self.company_name = self.__extract_value_from_tr_row(
            row,
            ("НГДУ", "Unnamed: 1_level_1", "Unnamed: 1_level_2", "Unnamed: 1_level_3"),
            "company_name",
        )
        self.subcompany_name = self.__extract_value_from_tr_row(
            row,
            (
                "Цех",
                "Unnamed: 112_level_1",
                "Unnamed: 112_level_2",
                "Unnamed: 112_level_3",
            ),
            "subcompany_name",
        )

        self.reservoir_name_str = self.__extract_value_from_tr_row(
            row,
            ("Пласт", "Unnamed: 7_level_1", "Unnamed: 7_level_2", "Unnamed: 7_level_3"),
            "reservoir_name_str",
        )

        self.field_name_str = self.__extract_value_from_tr_row(
            row,
            ("Месторождение", "Unnamed: 2_level_1", "Название", "Unnamed: 2_level_3"),
            "field_name_str",
        )
        self.well_type = self.__extract_value_from_tr_row(
            row,
            (
                "Тип\nскважины",
                "Unnamed: 5_level_1",
                "Unnamed: 5_level_2",
                "Unnamed: 5_level_3",
            ),
            "well_type",
        )

        self.d_cas_mm = self.__extract_value_from_tr_row(
            row, ("D э/к", "Unnamed: 9_level_1", "Unnamed: 9_level_2", "мм"), "d_cas_mm"
        )
        self.d_tube_mm = self.__extract_value_from_tr_row(
            row,
            ("D нкт", "Unnamed: 10_level_1", "Unnamed: 10_level_2", "мм"),
            "d_tube_mm",
        )

        self.d_choke_mm = self.__extract_value_from_tr_row(
            row,
            ("D шт", "Unnamed: 11_level_1", "Unnamed: 11_level_2", "мм"),
            "d_choke_mm",
        )
        if self.d_choke_mm == None:
            self.d_choke_mm = 8

        self.gamma_oil = self.__extract_value_from_tr_row(
            row,
            ("Плот-ть\nнефти", "Unnamed: 46_level_1", "Unnamed: 46_level_2", "г/см3"),
            "gamma_oil",
        )
        self.gamma_wat = self.__extract_value_from_tr_row(
            row,
            ("Плот-ть\nводы", "Unnamed: 47_level_1", "Unnamed: 47_level_2", "г/см3"),
            "gamma_wat",
        )
        self.gamma_gas = 0.7
        self.tres_c = self.__extract_value_from_tr_row(
            row, ("T пл", "Unnamed: 37_level_1", "Unnamed: 37_level_2", "ºC"), "tres_c"
        )
        self.pb_atm = self.__extract_value_from_tr_row(
            row,
            ("Р нас", "Unnamed: 35_level_1", "Unnamed: 35_level_2", "атм"),
            "pb_atm",
        )
        self.rsb_m3m3 = 213

        self.rp_m3m3 = self.__extract_value_from_tr_row(
            row, ("ГФ", "Unnamed: 36_level_1", "Unnamed: 36_level_2", "м3/т"), "rp_m3m3"
        )
        self.watercut_perc = self.__extract_value_from_tr_row(
            row,
            ("Фактический режим", "Обводненность", "Unnamed: 31_level_2", "%"),
            "watercut_perc",
        )
        self.muob_cp = self.__extract_value_from_tr_row(
            row,
            (
                "В-сть нефти\nв пл.\nусловиях ",
                "Unnamed: 42_level_1",
                "Unnamed: 42_level_2",
                "сПз",
            ),
            "muob_cp",
        )
        self.bob_m3m3 = self.__extract_value_from_tr_row(
            row,
            ("Об. к-т", "Unnamed: 45_level_1", "Unnamed: 45_level_2", "м3/м3"),
            "bob_m3m3",
        )

        self.well_work_mode = self.__extract_value_from_tr_row(
            row,
            (
                "Режим работы УЭЦН",
                "Режим",
                "Unnamed: 173_level_2",
                "Unnamed: 173_level_3",
            ),
            "well_work_mode",
        )

    def get_data_from_tr(self, tr_filename, well_name, field):
        print(f"Чтение техрежима {tr_filename}")
        try:
            tr = pd.read_excel(
                tr_filename, skiprows=6, header=[0, 1, 2, 3]
            )  # при ошибке файл нужно открыть и сохранить повторно без изменений
            print("Техрежим успешно прочитан")
        except:
            print("Ошибка, файл не найден")
            return None
        print(f"Поиск данных по скважине {well_name} месторождения {field}")
        this_well_row = tr[
            (
                    tr[
                        (
                            "№\nскв",
                            "Unnamed: 4_level_1",
                            "Unnamed: 4_level_2",
                            "Unnamed: 4_level_3",
                        )
                    ]
                    == well_name
            )
            & (
                    tr[
                        (
                            "Месторождение",
                            "Unnamed: 2_level_1",
                            "Название",
                            "Unnamed: 2_level_3",
                        )
                    ]
                    == field
            )
            ]
        self.tr_filename = tr_filename
        self.tr_dataframe = tr
        self.well_name = well_name
        if this_well_row.shape[0] == 0:
            print("Внимание, данных по скважине в техрежиме нет!")
            return None
        else:
            self.__fill_by_true_tr(this_well_row)

    def get_data_from_pvt_file(self, path, well_name):
        df = pd.read_pickle(path + well_name + "/pvt.pql", compression="gzip")

        self.gamma_oil = float(df.loc["gamma_oil_spl", well_name])
        self.gamma_gas = float(df.loc["gamma_gas_spl", well_name])
        self.gamma_wat = float(df.loc["gamma_wat_spl", well_name])
        self.rsb_m3m3 = float(df.loc["rsb_m3m3_spl", well_name])
        self.tres_c = float(df.loc["tres_c_spl", well_name])
        self.pb_atm = float(df.loc["pb_atma_spl", well_name])
        self.bob_m3m3 = float(df.loc["bob_m3m3_spl", well_name])
        self.muob_cp = float(df.loc["muob_cp_spl", well_name])
        self.rp_m3m3 = float(df.loc["rp_m3m3_sk", well_name])
        self.d_choke_mm = 12

    def get_data_from_pvt(self):
        (
            self.gamma_oil,
            self.gamma_gas,
            self.gamma_wat,
            self.rsb_m3m3,
            self.tres_c,
            self.pb_atm,
            self.bob_m3m3,
            self.muob_cp,
            self.rp_m3m3,
            self.d_choke_mm,
        ) = [x for x in self._list_with_pvt_params]

    def get_data_for_shelf(self):
        self.gamma_oil = 0.91
        self.gamma_gas = 0.71
        self.gamma_wat = 1.023
        self.rsb_m3m3 = 42.9
        self.tres_c = 58
        self.pb_atm = 120
        self.bob_m3m3 = 1.1398
        self.muob_cp = 2.77

    def __extract_value_from_tr_row(self, row, column_name, parameter_name):
        value = row[column_name].values[0]
        print(f"Для параметра {parameter_name} прочитано значение {value}")
        if pd.notna(value):
            return value
        else:
            return None

    def save_static_data_to_excel(self, static_excel_full_path):
        print("\n" + f"Сохранение статичных данных в {static_excel_full_path} \n")
        print("Итоговые статичные данные выглядят так")
        static_data_dict = {}
        for i in self.__dict__.items():
            if i[0] != "tr_dataframe":
                static_data_dict[i[0]] = i[1]
                print(i)

        static_data_df = pd.DataFrame(static_data_dict, index=[0])
        static_data_df = static_data_df.T
        static_data_df.to_excel(static_excel_full_path)
        print("Сохранение данных прошло успешно")

    def load_static_data(self, static_excel_full_path):
        print("\n" + f"Загрузка статичных данных из  {static_excel_full_path} \n")
        static_df = pd.read_excel(static_excel_full_path, index_col=0)

        self.company_name = static_df[0].company_name
        self.subcompany_name = static_df[0].subcompany_name
        self.well_name = static_df[0].well_name
        self.well_work_mode = static_df[0].well_work_mode
        self.well_type = static_df[0].well_type

        self.tr_filename = static_df[0].tr_filename
        self.tr_dataframe = None
        self.field_name_str = static_df[0].field_name_str
        self.reservoir_name_str = static_df[0].reservoir_name_str
        self.d_cas_mm = static_df[0].d_cas_mm
        self.d_tube_mm = static_df[0].d_tube_mm
        self.d_choke_mm = static_df[0].d_choke_mm

        self.gamma_oil = static_df[0].gamma_oil
        self.gamma_gas = static_df[0].gamma_gas
        self.gamma_wat = static_df[0].gamma_wat
        self.rsb_m3m3 = static_df[0].rsb_m3m3
        self.tres_c = static_df[0].tres_c
        self.pb_atm = static_df[0].pb_atm
        self.bob_m3m3 = static_df[0].bob_m3m3
        self.muob_cp = static_df[0].muob_cp
        self.rp_m3m3 = static_df[0].rp_m3m3

        self.qliq_m3day = static_df[0].qliq_m3day

        print("\nВыведем загруженные данные\n")
        for i in self.__dict__.items():
            if i[0] != "tr_dataframe":
                print(i)
        print("\nДанные успешно загружены")

    def load_wat_cut_hal(well_name):
        return pd.read_pickle(
            "../mess/" + well_name + "_wat_cut_volume.pql", compression="gzip"
        )
