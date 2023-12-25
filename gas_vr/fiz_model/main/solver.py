"""
Модуль для расчета физики
"""

import sys

import clr
import pandas as pd
import scipy

from global_names import GlobalNames
from static_names import StaticData
from vfm_settings import VFM_Settings

gn = GlobalNames()


class Solver:
    """
    Класс для расчета штуцера
    """

    def __init__(
            self,
            data: pd.DataFrame,
            settings: VFM_Settings,
            q_column_to_use: str,
            list_with_pvt_params: list,
            field: str,
            engine: str,
    ):
        """
        :param data: отфильтрованные данные
        :param settings: Настройки
        :param q_column_to_use: название колонки с таргетом
        :param list_with_pvt_params: лист с pvt параметрами
        :param field: название месторождения
        :param engine: Движок
        """
        # инициализация переменных, подаваемых на вход Solver
        self.data = data
        self.settings = settings
        self.q_column_to_use = q_column_to_use
        self._list_with_pvt_params = list_with_pvt_params
        self._field = field
        self.static_data = StaticData()

        self.restored_df = None
        self.q_gas_m3day_choke = None
        self.get_static_class_inst()
        self.get_median_values()

        if engine == "unifloc_net":
            sys.path.append("../unifloc_net/")
            clr.AddReference("alglibnet2")
            clr.AddReference("UnfClassLibrary")
            clr.AddReference("u7_excel")
            clr.AddReference("Newtonsoft.Json")

        self.pvt_calc(engine)
        self.calibr_calc(engine)

    @property
    def list_with_pvt_params(self):
        """
        Свойство получения списка PVT параметров
        """
        return self._list_with_pvt_params

    @list_with_pvt_params.setter
    def list_with_pvt_params(self, list_):
        """
        Свойство задания списка PVT параметров
        """
        self._list_with_pvt_params = list_

    @property
    def field(self):
        """
        Свойство получения месторождения
        """
        return self._field

    @field.setter
    def field(self, field):
        """
        Свойство задания месторождения
        """
        self._field = field

    def get_static_class_inst(self):
        """
        Метод заполнения атрибута static data данными с pvt листа
        """
        self.static_data._list_with_pvt_params = self._list_with_pvt_params
        self.static_data.get_data_from_pvt()

    def get_median_values(self, wat_cut_type="tm"):
        """
        Метод получения медианных значений для некоторых параметров

        Parameters
        ----------
        :param wat_cut_type: отвечает за обводненность, которую необходимо
                             осреднить для дальнейшего использования -
                             'approved' - утвержденная;
                             'hal' - ХАЛ;
                             'tm' - замерная
        """
        if self.q_column_to_use == gn.q_liq_m3day_VX:
            self.static_data.get_median_values_from_VX_data(self.data)
        else:
            self.static_data.get_median_values_from_TM_data(self.data, wat_cut_type)

    def pvt_calc(self, engine: str):
        """
        Метод расчета pvt строки

        Parameters
        ----------
        :param engine: двигатель расчета,
                       'unifloc_net';
                       'unifloc_py';
                       'unifloc_vba'
        """
        if engine == "unifloc_net":
            import u7_excel

            self.pvt_str = u7_excel.u7_Excel_function_servise.PVT_encode_string(
                gamma_gas=self.static_data.gamma_gas,
                gamma_oil=self.static_data.gamma_oil,
                gamma_wat=self.static_data.gamma_wat,
                rsb_m3m3=self.static_data.rsb_m3m3,
                rp_m3m3=self.static_data.gor_median,
                pb_atma=self.static_data.pb_atm,
                bob_m3m3=self.static_data.bob_m3m3,
                muob_cP=self.static_data.muob_cp,
            )
        elif engine == "unifloc_vba":
            sys.path.append("../unifloc_vba/unifloc_vba_7_25")
            import python_api

            UniflocVBA = python_api.API(
                "../unifloc_vba/unifloc_vba_7_25/UniflocVBA_7.xlam"
            )

            self.pvt_str = UniflocVBA.PVT_encode_string(
                gamma_gas=self.static_data.gamma_gas,
                gamma_oil=self.static_data.gamma_oil,
                gamma_wat=self.static_data.gamma_wat,
                rsb_m3m3=self.static_data.rsb_m3m3,
                rp_m3m3=self.static_data.gor_median,
                pb_atma=self.static_data.pb_atm,
                bob_m3m3=self.static_data.bob_m3m3,
                muob_cP=self.static_data.muob_cp,
            )

    @staticmethod
    def calc_core(
            q_liq: float,
            this_row: pd.DataFrame,
            pvt_str: str,
            static_data: StaticData,
            engine: str,
    ):
        """
        Метод расчета калибровочного коэффициента

        Parameters
        ----------
        :param q_liq: Дебит жидкости, м3/сут
        :param this_row: единичная строка numpy из DataFrame
        :param pvt_str: pvt строка
        :param static_data: экземпляр класса статических данных с готовым
                            набором pvt свойств и осредненных параметров
        :param engine: двигатель расчета,
                       'unifloc_net';
                       'unifloc_py';
                       'unifloc_vba'

        :return: объект, содержащий калибровку по штуцеру
        """
        if engine == "unifloc_net":
            import u7_excel

            return u7_excel.u7_Excel_functions_MF.MF_calibr_choke_fast(
                qliq_sm3day=q_liq,
                fw_perc=static_data.watercut_perc_median,
                d_choke_mm=this_row[gn.d_choke_mm].values[0],
                p_in_atma=this_row[gn.p_buf_atm].values[0],
                p_out_atma=this_row[gn.p_lin_atm].values[0],
                d_pipe_mm=65,
                t_choke_C=static_data.t_lin_median,
                q_gas_sm3day=0,
                str_PVT=pvt_str,
            )
        elif engine == "unifloc_vba":
            sys.path.append("../unifloc_vba/unifloc_vba_7_25")
            import python_api

            UniflocVBA = python_api.API(
                "../unifloc_vba/unifloc_vba_7_25/UniflocVBA_7.xlam"
            )
            return UniflocVBA.MF_calibr_choke_fast(
                qliq_sm3day=q_liq,
                fw_perc=static_data.watercut_perc_median,
                d_choke_mm=this_row[gn.d_choke_mm].values[0],
                p_in_atma=this_row[gn.p_buf_atm].values[0],
                p_out_atma=this_row[gn.p_lin_atm].values[0],
                d_pipe_mm=65,
                t_choke_C=static_data.t_lin_median,
                q_gas_sm3day=0,
                str_PVT=pvt_str,
            )

    @staticmethod
    def cut_tuple(data: object, engine: str) -> float:
        """
        Метод обработки результатов расчета с разных двигателей

        :param data:результат расчета unifloc
        :param engine: двигатель расчета,
                       'unifloc_net';
                       'unifloc_py';
                       'unifloc_vba'

        :return:обработанное значение с плавающей точкой
        """
        if engine == "unifloc_net":
            changed_data = data[0]
        elif engine == "unifloc_vba":
            changed_data = data[0][0]

        return changed_data

    def calibr_calc(self, engine: str):
        """
        Метод расчета и добавления данных по калибровочному коэффициенту в основной дата сет

        Parameters
        ----------
        :param engine: двигатель расчета,
                       'unifloc_net';
                       'unifloc_py';
                       'unifloc_vba'
        """
        choke_calibr_d = []
        for i in range(self.data.shape[0]):
            this_row = self.data[self.data.index == self.data.index.values[i]]
            this_liq_rate = this_row[self.q_column_to_use].values[0]

            this_choke_calibr_d = Solver.calc_core(
                this_liq_rate, this_row, self.pvt_str, self.static_data, engine
            )
            if this_choke_calibr_d == -1:
                this_choke_calibr_d = -1
            else:
                this_choke_calibr_d = Solver.cut_tuple(
                    data=this_choke_calibr_d, engine=engine
                )

            choke_calibr_d.append(this_choke_calibr_d)

        self.data[gn.c_calibr_choke] = choke_calibr_d
        self.data = self.data.dropna(subset=[gn.c_calibr_choke])

    @staticmethod
    def wrapper_for_minimize(
            q_liq: float,
            interpolated_calibr: float,
            this_row,
            static_data: StaticData,
            pvt_str,
            engine: str,
    ) -> float:
        """
        Метод оптимизации солвера для решения обратной задачи по нахождению дебита.
        Используется как внутренняя функция при расчете дебита.
        Работает с единичным набором параметров

        Parameters
        ----------
        :param q_liq: Дебит жидкости, м3/сут
        :param interpolated_calibr: Калибровочный коэффиент для текущего набора параметров
        :param this_row: единичная строка numpy из DataFrame
        :param pvt_str: pvt строка
        :param static_data: экземпляр класса статических данных с готовым
                            набором pvt свойств и осредненных параметров
        :param engine: двигатель расчета,
                       'unifloc_net';
                       'unifloc_py';
                       'unifloc_vba'

        :return: найденное значение дебита после оптимизатора

        """
        vba_result = Solver.calc_core(q_liq[0], this_row, pvt_str, static_data, engine)

        if vba_result == -1:
            vba_result = -1
        else:
            vba_result = Solver.cut_tuple(data=vba_result, engine=engine)
        try:  # непонятно, почему появляются nan
            (vba_result - interpolated_calibr) ** 2
        except:
            vba_result = -1

        return (vba_result - interpolated_calibr) ** 2

    def q_gas_calc(self, result_df: pd.DataFrame, target: str, engine: str):
        """
        Итоговая функция по расчету дебита при решении обратной задачи

        Parameters
        ----------
        :param result_df: общий датафрейм с предсказанной калибровкой
        :param target: Наименование таргета
        :param engine: двигатель расчета,
                       'unifloc_net';
                       'unifloc_py';
                       'unifloc_vba
        """
        df_with_restored_target = result_df.dropna(subset=[target + gn.add_restore])
        self.q_gas_m3day_choke = {
            row: [] for row in self.settings.models
        }
        for i in range(df_with_restored_target.shape[0]):
            this_row = df_with_restored_target[
                df_with_restored_target.index == df_with_restored_target.index.values[i]
            ]
            for this_model in self.q_gas_m3day_choke.keys():
                this_q_gas_m3day_choke = scipy.optimize.minimize(
                    Solver.wrapper_for_minimize,
                    20,
                    args=(
                        this_row[target + gn.add_restore + this_model].values[0],
                        this_row,
                        self.static_data,
                        self.pvt_str,
                        engine,
                    ),
                    method="BFGS",
                ).x[0]
                if this_q_gas_m3day_choke == 20:
                    this_q_gas_m3day_choke = 0
                self.q_gas_m3day_choke[this_model].append(this_q_gas_m3day_choke)
