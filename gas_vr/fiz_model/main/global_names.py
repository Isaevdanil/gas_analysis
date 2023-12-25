class GlobalNames:
    """Класс для хранения глобальных переменных.

    Используется для удобства работы с данными.
    """

    def __init__(self):
        # общие параметры для всех скважин
        self.q_liq_m3day = "Дебит жидкости, м3/сут"
        self.q_liq_mass_tday = "Дебит жидкости массовый, т/сут"
        self.q_gas_m3day = "Дебит газа, м3/сут"
        self.q_wat_m3day = "Дебит воды, м3/сут"
        self.q_oil_m3day = "Дебит нефти, м3/сут"
        self.q_oil_mass_tday = "Дебит нефти массовый, т/сут"

        self.watercut_perc = "Обводненность, %"
        self.watercut_perc_hal = "Обводненность, % ХАЛ"
        self.watercut_perc_approved = "Обводненность, % УТВ"
        self.dens_liq = "Плотность жидкости, кг/м3"
        self.dens_gas = "Плотность газа, кг/м3"
        self.gor_m3m3 = "Газовый фактор, м3/м3"
        self.wor_m3m3 = "Газожидкостной фактор, м3м3"
        self.t_fluid_c = "Температура жидкости, C"
        self.t_in_sep = "Температура в сепараторе, С"

        self.p_buf_atm = "Буферное давление, атм"
        self.p_lin_atm = "Линейное давление, атм"
        self.p_on_gzu = "Давление на ГЗУ, атм"
        self.p_in_ag_line = "Давление в линии АГ, атм"
        self.p_cas_atm = "Затрубное давление, атм"

        self.well_name = "Номер скважины"
        self.time = "Время"
        self.well_id = "id скважины"
        self.param_name = "Наименование параметра"
        self.param_value = "Значение параметра"
        self.param_id = "id параметра"
        self.tm_chess_time = "Tm_chess"
        self.recovery_type = "Способ эксплуатации"
        # для фонтана
        self.dp_atm = "Перепад давления на штуцере, атм"
        self.d_choke_mm = "Диаметр штуцера, мм"
        self.c_calibr_choke = "К. калибровка по штуцеру, ед"

        # для газлифта
        self.gas_flow_rate_calc = "Расход газа при стандартных условиях (косв. расчёт)"
        self.gas_flow_rate_meas = "Расход газа при стандартных условиях (по счётчику)"
        self.p_after_choke = "Давление после клапана, атм"
        self.t_after_choke = "Температура после клапана, С"
        self.p_before_choke = "Давление до клапана, атм"
        self.t_before_choke = "Температура до клапана, С"

        # для данных VX и Roxar
        self.p_lin_atm_VX = "Давление линейное, атм vx"
        self.t_lin_C_VX = "Температура линейная, С vx"
        self.q_oil_mass_tday_VX = "Дебит нефти массовый, т/сут vx"
        self.q_gas_m3day_VX = "Дебит газа, ст. м3/сут vx"
        self.q_liq_m3day_VX = "Дебит жидкости, м3/сут vx"
        self.watercut_perc_VX = "Обводненность, % vx"
        self.gor_m3t_VX = "Газовый фактор (на тонну нефти), м3/т vx"
        self.gor_m3m3_VX = "Газовый фактор (на м3 нефти), м3/м3 vx"
        self.q_liq_accumulated_VX = "Дебит жидкости накопленный, м3 vx"

        # для ЭЦН
        self.freq_Hz = "Частота, Гц"
        self.active_power_kW = "Активная мощность, кВ"
        self.active_power_calc_kW = "Активная мощность (расч), кВ"
        self.ped_t = "Температура двигателя, С"
        self.p_intake_esp_atm = "Давление на приеме, атм"
        self.p_dis_esp_atm = "Давление на выкиде, атм"
        self.volt_CA_v = "Входное напряжение СА, В"
        self.dp_esp_atm = "Перепад давления в насосе, атм"
        self.t_intake_c = "Температура на приеме, С"
        self.t_dis_c = "Температура на выкиде, С"

        # Общее
        self.add_restore = " restore"
        self.abs_error = "absolute error"
        self.relative_error = "relative error"
        self.add_lr = " linear regression"
        self.add_ridge = " ridge regression"
        self.add_lasso = " lasso regression"
        self.add_decision_tree = " decision tree"
        self.add_random_forest = " random forest"

    def return_dict_column_to_rename(self):
        """
        функция для переименования столбцов таблицы
        при поступлении новых данных можно обновлять словарь
        """
        columns_name_to_rename = {
            self.q_liq_m3day: [
                "Объемный дебит жидкости",
                "Дебит жидкости (ТМ)",
                "Дебит жидкости (ТМ)",
                "LIQ_RATE",
                "Дебит жидкости Qж (м3/сут)",
            ],
            self.q_liq_mass_tday: ["Дебит жидкости массовый (ТМ)"],
            self.q_gas_m3day: [
                "Объемный дебит газа",
                "Дебит газа (ТМ)",
                "QGAS",
                "Дебит газа Qг (м3/сут)",
            ],
            self.q_oil_mass_tday: [
                "Дебит нефти (ТМ)",
                "Дебит нефти массовый, т/сут",
                "OIL_RATE",
            ],
            self.q_oil_m3day: ["Дебит нефти Qн (м3/сут)"],
            self.watercut_perc: [
                "Процент обводненности",
                "Обводненность (ТМ)",
                "Обв",
                "WATER_CUT",
                "Обводненность, %",
            ],
            self.dens_liq: ["Плотность жидкости (ТМ )"],
            self.dens_gas: ["Плотность газа (ТМ )"],
            self.gor_m3m3: ["ГФР(ТМ)", "ГФ, м3/т", "Газовый фактор (рассчитанный)"],
            self.wor_m3m3: ["Газожидкостной фактор (рассчитанный)"],
            self.t_fluid_c: [
                "Температура жидкости (ТМ)",
                "Линейная температура, С",
                "Температура линейная, С",
            ],
            self.t_in_sep: [
                "Температура в сепараторе ТМ",
                "Температура в сепараторе (ТМ)",
            ],
            self.p_buf_atm: [
                "Давление буферное (ТМ)",
                "Устьевое давление, (ат)",
                "Давление буферное, атм",
            ],
            self.p_lin_atm: [
                "Линейное давление",
                "Давление линейное (ТМ)",
                "Давление линейное (ТМ)",
                "PLIN",
                "Линейное давление, (ат)",
                "Давление линейное, атм",
            ],
            self.p_on_gzu: ["Давление на ГЗУ (ТМ)", "Давление в коллекторе ГЗУ"],
            self.p_in_ag_line: ["Давление в линии АГ (ТМ)"],
            self.p_cas_atm: [
                "Давление затрубное (ТМ)",
                "Затрубное давление, (ат)",
                "Давление затрубное, атм",
            ],
            self.dp_atm: ["Перепад давления на штуцере, атм"],
            self.well_name: ["Well name", "WELL_NAME", "№ Скв"],
            self.time: ["Date time", "DT", "Дата", "Дата.1", "Дата.2"],
            self.well_id: ["Well_ID", "Well id", "WELL_ID", "SK_1"],
            self.param_name: ["LONG_NAME", "Long name"],
            self.param_value: ["VAL", "Value"],
            self.param_id: ["PARAM_ID"],
            self.tm_chess_time: ["TM_CHESS_INSERT", "Tm chess insert DT"],
            self.recovery_type: ["СЭ", "SE_NAME"],
            self.gas_flow_rate_calc: [
                "Расход газа при стандартных условиях (косв. расчёт) (ТМ)"
            ],
            self.gas_flow_rate_meas: [
                "Расход газа при стандартных условиях (по счётчику) (ТМ)",
                "Расход АГ (ТМ)",
            ],
            self.p_after_choke: ["Давление после клапана (ТМ)"],
            self.t_after_choke: ["Температура после клапана (ТМ)"],
            self.p_before_choke: ["Давление до клапана (ТМ)"],
            self.t_before_choke: ["Температура до клапана (ТМ)"],
            self.freq_Hz: ["FREQ_HZ"],
            self.active_power_kW: ["ACTIV_POWER", "Активная мощность, (кВт)"],
            self.active_power_calc_kW: ["ACTIV_POWER_CALC"],
            self.ped_t: ["PED_T"],
            self.p_intake_esp_atm: [
                "PINP",
                "Давление на пр-ме нас. (пласт. жидк.), (ат)",
                "Давление на приеме, атм",
            ],
            self.p_dis_esp_atm: ["Давление на выкиде, атм"],
            self.dp_esp_atm: ["Перепад давления в насосе, атм"],
        }

        return columns_name_to_rename

    def rename_vx_cols(self):
        """
        функция перезаписывает глобальные имена при использовании данных с VX или Roxar
        """
        self.p_lin_atm_VX = self.p_lin_atm
        self.t_lin_C_VX = self.t_fluid_c
        self.q_oil_mass_tday_VX = self.q_oil_mass_tday
        self.q_gas_m3day_VX = self.q_gas_m3day
        self.q_liq_m3day_VX = self.q_liq_m3day
        self.watercut_perc_VX = self.watercut_perc
        self.gor_m3t_VX = None
        self.gor_m3m3_VX = self.gor_m3m3
        self.q_liq_accumulated_VX = None
