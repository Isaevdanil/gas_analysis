"""Модуль для описания основных настроек для расчета."""
import datetime
import json
import os

import pandas as pd

from global_names import GlobalNames

gn = GlobalNames()


class VFM_Settings:
    """Класс для хранения основных настроек расчета для ВР."""

    def __init__(self):
        """
        При инициализации класса агружаем файл json с основными настройками

        :param field: Месторожедние
        :param wells: Номер скважины
        :param path_to_tr: Путь до данных ТР
        :param path_to_save_root: Путь для сохранения артефактов
        :param path_to_all_data: Путь для сохранения артефактов
        :param resample_time: Время осреднения параметов
        :param target: Целевая переменная на этапе адаптации
        :param predict: Целевая переменная на этапе восстановления дебита
        :param date_for_calc: Временной промежуток для анализа
        :param engine: Движок
        :param full_path_to_save: Полный путь для сохранения артефактов
        :param full_path_to_all_data: Полный путь для сохранения артефактов
        :param plot: Флаг построения графиков
        :param models: Список ML моделей
        """
        with open("vfm_settings.json", "r", encoding="utf-8") as read_file:
            settings_dict = json.load(read_file)

        (
            self.field,
            self.wells,
            self.path_to_tr,
            self.path_to_save_root,
            self.path_to_all_data,
            self.resample_time,
            self.target,
            self.predict,
            self.date_for_calc,
            self.engine,
            self.plot,
            self.models
        ) = (x for x in settings_dict.values())

        self.full_path_to_save = self.path_to_save_root + self.field + "/"
        self.full_path_to_all_data = self.path_to_all_data + self.field + "/"
        self.prepare_ml_model_list()
        self._well_name = None

    def prepare_ml_model_list(self):
        """
        Метод для обработки списка МЛ моделей для расчета калибровочных коэффициентов.
        """
        models = [""]
        for this_model in self.models:
            if this_model == "lr":
                models.append(gn.add_lr)
            if this_model == "ridge":
                models.append(gn.add_ridge)
            if this_model == "lasso":
                models.append(gn.add_lasso)
            if this_model == "rf":
                models.append(gn.add_random_forest)
        self.models = models

    def todays_date_folder(self) -> str:
        """
        Метод для создание папки с текущей датой
        для сохранения результатов

        :return: Путь к папке, где будут сохраняться результаты
        """
        today_date = str(datetime.datetime.now().date())
        today_date = today_date.replace("-", ".")
        try:
            os.mkdir(self.full_path_to_save + today_date)
            os.mkdir(self.full_path_to_save + today_date + "/" + self._well_name)
        except:
            pass

        return self.full_path_to_save + today_date + "/"

    @staticmethod
    def check_q_gas_data(q_column_to_use: str, data: pd.DataFrame) -> str:
        """
        Метод для определения типа данных по замерам (интегральные или дифференциальные замеры)

        Parameters
        ----------
        :param q_column_to_use: колонка с наименованием дебита - таргета
        :param data: датафрейм с данными

        :return: Строка - определитель типа данных
        """
        if q_column_to_use == gn.q_gas_m3day:
            this_data = data.dropna(subset=[q_column_to_use])
            short_time, long_time = 0, 0
            for i in range(this_data.shape[0] - 1):
                if (this_data.index[i + 1] - this_data.index[i]).total_seconds() < 300:
                    short_time += 1
                else:
                    long_time += 1
            if long_time > short_time:
                print("Единичные замеры с АГЗУ")
                return "one"
            else:
                print("Дифференциальные замеры с АГЗУ")
                return "many"
        else:
            return "many"
