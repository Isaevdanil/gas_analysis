import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from sklearn import linear_model, dummy, ensemble
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from global_names import GlobalNames
from vfm_settings import VFM_Settings

gn = GlobalNames()


class ML:
    """
    Класс для работы с ML алгоритмами
    """

    def __init__(
            self,
            field: str,
            well_name: str,
            settings: VFM_Settings,
            df_with_liq_rate: pd.DataFrame,
            df_without_liq_rate: pd.DataFrame,
            q_data_type: str,
            target: str,
            predict: str,
            transform_type: str,
            q_column_to_use: str,
    ):
        """

        Parameters
        ----------
        :param: field: Данные
        :param well_name: Номер скважины в виде строки
        :param settings: Настройки
        :param df_with_liq_rate: Дата сет, содержащий исключительно точки, в которых есть замеры дебита
        :param df_without_liq_rate: Дата сет без замерных параметров
        :param q_data_type: Тип данных по замерам:
                            many - дифференциальные замеры (в основном раз в минуту)
                            one - осредненные замеры
        :param target: наименование колонки с таргетом
        :param predict: флаг на тестирование/восстановление,
                        0 - тестирование
                        1 - восстановление на заданном промежутке
        :param transform_type: тип трансформации данных для ML,
                              standartization, normalization, None
                              В большистве случаев рекомеендуется использовать standartization
        :param q_column_to_use: наименование колонки с дебитом, который необходимо восстановить
        """
        self.field = field
        self.well_name = well_name
        self.settings = settings
        self.df_with_liq_rate = df_with_liq_rate
        self.df_without_liq_rate = df_without_liq_rate
        self.q_data_type = q_data_type
        self.target = target
        self.predict = predict
        self.transform_type = transform_type
        self.q_column_to_use = q_column_to_use
        self.df_with_predicted_calibr = None
        self.test_result_df = None
        self.error_calibr_df = None

        self.restore_target()

    @staticmethod
    def select_features(
            df_with_liq_rate: pd.DataFrame,
            df_without_liq_rate: pd.DataFrame,
            target: str,
            predict: str,
    ) -> tuple:
        """
        Метод выбора нужных колонок в качестве параметров для обучения

        :param df_with_liq_rate: Дата сет, содержащий исключительно точки, в которых есть замеры дебита
        :param df_without_liq_rate: Дата сет без замерных параметров
        :param target: наименование колонки с таргетом
        :param predict: флаг на тестирование/восстановление,
                        0 - тестирование
                        1 - восстановление на заданном промежутке

        :return: кортеж с 2 датафреймами: 1-с точками, в которых есть замеры дебита, 2-без замеров дебита
        """
        features_list = [gn.dp_atm, gn.d_choke_mm, gn.p_lin_atm, gn.p_buf_atm]
        cols_to_save = features_list + [target]
        new_df = pd.DataFrame(df_with_liq_rate, columns=cols_to_save)
        new_df_without_oil_rate = None

        if predict == 1:
            new_df_without_oil_rate = pd.DataFrame(
                df_without_liq_rate,
                index=df_without_liq_rate.index,
                columns=features_list,
            )
            new_df_without_oil_rate = new_df_without_oil_rate.dropna(
                subset=features_list
            )

        return new_df, new_df_without_oil_rate

    @staticmethod
    def get_test_train_drop_2_points(df: pd.DataFrame) -> tuple:
        """
        Метод разделения данных 50/50

        :param df: дата сет, который необходимо разделить
        :return: кортеж с двумя датафреймами - для обчучения и для тестирования
        """
        if type(df) is pd.DataFrame:
            df = df.reset_index()
            print(df)
            train_data, test_data = df[df.index % 2 == 0].set_index("index"), df[
                ~(df.index % 2 == 0)
            ].set_index("index")
        else:
            train_data, test_data = (
                df[pd.DataFrame(df).index % 2 == 0],
                df[~(pd.DataFrame(df).index % 2 == 0)],
            )

        return train_data, test_data

    @staticmethod
    def x_y_sep(df: pd.DataFrame, target: str) -> tuple:
        """
        Метод разделения фичей и таргета

        :param df: исходный дата сет с X и y
        :param target: наименование колонки с таргетом (y)

        :return: кортеж с двумя датафреймами: 1-фичи, 2-таргеты
        """
        y = df[[target]]
        x = df.drop(columns=[target])
        return x, y

    @staticmethod
    def new_features(
            df_with_liq_rate: pd.DataFrame,
            df_without_liq_rate: pd.DataFrame,
            predict: str,
            target: str,
    ) -> tuple:
        """
        Метод подготовки данных для ml алгоритма. Создает дополнительные фичи и
        разделяет данные на тестовую и обучающую выборки

        :param df_with_liq_rate: Дата сет, содержащий исключительно точки, в которых есть замеры дебита
        :param df_without_liq_rate: Дата сет без замерных параметров
        :param predict: флаг на тестирование/восстановление,
                        0 - тестирование
                        1 - восстановление на заданном промежутке
        :param target: наименование колонки с таргетом
        :param create_method: метод создания дополнительных фичей
                              None - дополнительные фичи не создаются

        :return: кортеж с фичами, таргетам и дополнительно фичами для полного восстановления
        """
        list_with_2_dfs = ML.select_features(
            df_with_liq_rate, df_without_liq_rate, target, predict
        )
        X, y = ML.x_y_sep(list_with_2_dfs[0], target)
        X, y = X, y
        X_test_for_predict = list_with_2_dfs[1]

        return X, y, X_test_for_predict

    @staticmethod
    def get_index_from_data(
            df_with_liq_rate: pd.DataFrame,
            df_without_liq_rate: pd.DataFrame,
            q_data_type: str,
            predict: str,
            target: str,
    ) -> tuple:
        """
        Метод получения индексов для тестовой и обучающей выборки

        :param df_with_liq_rate: Дата сет, содержащий исключительно точки, в которых есть замеры дебита
        :param df_without_liq_rate: Дата сет без замерных параметров
        :param q_data_type: Тип данных по замерам:
                            many - дифференциальные замеры (в основном раз в минуту)
                            one - осредненные замеры
        :param predict: флаг на тестирование/восстановление,
                        0 - тестирование
                        1 - восстановление на заданном промежутке
        :param target: наименование колонки с таргетом

        :return: кортеж: 1 - индексы для обучающей выборки
                         2 - индексы для тестовой выборки
        """
        data_list = ML.select_features(
            df_with_liq_rate, df_without_liq_rate, target, predict
        )
        if predict == 0:
            data = data_list[0]
            if q_data_type == "one":
                train, test = ML.get_test_train_drop_2_points(data)
            elif q_data_type == "many":
                train, test = train_test_split(
                    data, test_size=0.2, random_state=42, shuffle=False
                )
            train_index, test_index = train.index, test.index
        else:
            train_index, test_index = data_list[0].index, data_list[1].index

        return train_index, test_index

    def get_metrics_result(
            self, well_name: str, data: pd.DataFrame, target: str
    ) -> pd.DataFrame:
        """
        Метод получения метрик расчета

        :param well_name: номер скважны в виде строки
        :param y_test: массив с фактическими данными таргета
        :param prediction: массив с предсказанными данными таргета
        :param restore_q_liq: флаг для выбора наименований колонок
                              0 - метрики для калибровки
                              1 - метрики для дебита

        :return: Датафрейм с метриками по текущему расчету
        """

        df_metrics = data.dropna(subset=[target]).copy()

        for this_model in self.settings.models:
            df_metrics[gn.abs_error + this_model] = (
                    df_metrics[target] - df_metrics[target + gn.add_restore + this_model]
            )

            df_metrics[gn.relative_error + this_model] = (
                                                                 df_metrics[gn.abs_error + this_model] / df_metrics[
                                                             target]
                                                         ).abs() * 100

            y_test = df_metrics[target]
            prediction = df_metrics[target + gn.add_restore + this_model]

            df_metrics["R2" + this_model] = metrics.r2_score(y_test, prediction)
            df_metrics["MSE" + this_model] = metrics.mean_squared_error(y_test, prediction)
            df_metrics["MAE" + this_model] = metrics.mean_absolute_error(y_test, prediction)
            df_metrics["MAPE" + this_model] = metrics.mean_absolute_percentage_error(y_test, prediction)

        df_metrics["well_name"] = well_name

        return df_metrics

    @staticmethod
    def func_for_transform_data(
            x_1: np.array, x_2: np.array, transform_type=None
    ) -> tuple:
        """
        Метод преобразования данных для ml алгоритма

        :param x_1: фичи из обучающей выборки
        :param x_2: фичи из тестовой выборки
        :param transform_type: тип трансформации данных для ML,
                              standartization, normalization, None
                              В большистве случаев рекомеендуется использовать standartization

        :return: кортеж с преобразовынными данными x_1 и x_2
        """
        if transform_type == "standartization":
            scaler = preprocessing.StandardScaler().fit(x_1)
            x_1_scaled = scaler.transform(x_1)
            x_2_scaled = scaler.transform(x_2)
        elif transform_type == "normalization":
            x_1_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
                x_1
            )
            x_2_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
                x_2
            )
        elif transform_type == None:
            x_1_scaled = x_1
            x_2_scaled = x_2

        return x_1_scaled, x_2_scaled

    def restore_target(self):
        """
        Метод восстановления таргета
        """

        self.df_with_predicted_calibr = self.df_without_liq_rate.copy()
        self.df_with_predicted_calibr[self.target] = self.df_with_liq_rate[self.target]

        X, y, X_test_for_predict = self.new_features(
            df_with_liq_rate=self.df_with_liq_rate,
            df_without_liq_rate=self.df_without_liq_rate,
            predict=self.predict,
            target=self.target,
        )
        X_train, X_test, y_train = X.values, X_test_for_predict, y.values

        for this_model in self.settings.models:
            if this_model == "":
                model = dummy.DummyRegressor(strategy="median")
            if this_model == gn.add_lr:
                model = linear_model.LinearRegression()
            if this_model == gn.add_ridge:
                model = linear_model.RidgeCV()
            if this_model == gn.add_lasso:
                model = linear_model.LassoCV()
            if this_model == gn.add_random_forest:
                model = ensemble.RandomForestRegressor()

            model.fit(X_train, y_train.ravel())
            df_prediction = pd.DataFrame(
                model.predict(X_test.values),
                index=X_test.index,
                columns=[gn.c_calibr_choke + gn.add_restore + this_model]
            )

            self.df_with_predicted_calibr = self.df_with_predicted_calibr.join(
                [
                    df_prediction,
                ],
                how="left",
            )

            df_metrics = self.df_with_predicted_calibr[[
                gn.c_calibr_choke,
                gn.c_calibr_choke + gn.add_restore + this_model
            ]].dropna()

            y_test = df_metrics[gn.c_calibr_choke]
            prediction = df_metrics[gn.c_calibr_choke + gn.add_restore + this_model]

            print(gn.c_calibr_choke + gn.add_restore + this_model, "R2 =", metrics.r2_score(y_test, prediction))
            print(gn.c_calibr_choke + gn.add_restore + this_model, "MSE =",
                  metrics.mean_squared_error(y_test, prediction))
            print(gn.c_calibr_choke + gn.add_restore + this_model, "MAE =",
                  metrics.mean_absolute_error(y_test, prediction))
            print(gn.c_calibr_choke + gn.add_restore + this_model, "MAPE =",
                  metrics.mean_absolute_percentage_error(y_test, prediction))

        if self.settings.plot:
            self.plot_after_adaptation()

    def plot_after_adaptation(self):
        """
        Метод для построения графиков
        """
        data = self.df_with_predicted_calibr
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Дебит газа, м3/сут",
                "Калибровочные коэффициенты, ед.",
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

        for this_col in [
            gn.c_calibr_choke,
            gn.c_calibr_choke + gn.add_restore,
            gn.c_calibr_choke + gn.add_restore + gn.add_lr,
            gn.c_calibr_choke + gn.add_restore + gn.add_ridge,
            gn.c_calibr_choke + gn.add_restore + gn.add_lasso,
            gn.c_calibr_choke + gn.add_restore + gn.add_decision_tree,
            gn.c_calibr_choke + gn.add_restore + gn.add_random_forest,
        ]:
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
                    row=3,
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
                    row=4,
                    col=1,
                )

        fig.update_layout(
            title_text=f"Field={self.field}, Well={self.well_name}, Step=Adaptation",
            height=450 * 3,
        )

        fig.layout.hovermode = "x"

        plot(
            fig,
            filename=f"{self.settings.full_path_to_save}/{self.well_name}_after_adaptation_view.html",
            auto_open=True,
        )

    def restore_q_gas(self, q_gas_m3day_choke: list):
        """
        Метод получения результатов и построения графиков по текущему восстановлению дебита

        :param q_gas_m3day_choke: лист с восстановленными значениями дебита
        """
        df_with_restored_target = self.df_with_predicted_calibr.dropna(
            subset=[self.target + gn.add_restore]
        )
        df_with_predicted_q = pd.DataFrame(
            q_gas_m3day_choke,
            index=df_with_restored_target.index,
        )
        df_with_predicted_q.columns = [gn.q_gas_m3day + gn.add_restore + row for row in df_with_predicted_q.columns]
        self.test_result_df = pd.concat(
            [self.df_with_predicted_calibr, df_with_predicted_q], axis=1
        )

        if self.predict == 1:
            self.forecast_result_df = pd.concat(
                [self.test_result_df, self.df_with_liq_rate[self.q_column_to_use]],
                axis=1,
            )
            df_for_calc_meatrics = df_with_predicted_q.join(
                df_with_restored_target[
                    [
                        self.q_column_to_use,
                        gn.p_buf_atm,
                        gn.p_lin_atm,
                        gn.d_choke_mm,
                        gn.c_calibr_choke,
                        gn.c_calibr_choke + gn.add_restore,
                    ]
                ]
            ).dropna()
            self.error_qliq_df = self.get_metrics_result(
                self.well_name, df_for_calc_meatrics, target=self.q_column_to_use
            )
            if self.settings.plot:
                self.plot_after_prediction()

    def plot_after_prediction(self):
        """
        Метод для построения графиков
        """
        data = self.test_result_df
        df = self.error_qliq_df.round(2)
        fig = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=(
                "Метрики",
                "Дебит газа, м3/сут",
                "Калибровочные коэффициенты, ед.",
                "Давления, атм",
                "Диаметр штуцера, мм",
            ),
            shared_xaxes=True,
            vertical_spacing=0.01,
            specs=[
                [{"type": "table"}],
                [{"type": "scattergl"}],
                [{"type": "scattergl"}],
                [{"type": "scattergl"}],
                [{"type": "scattergl"}],
            ],
        )

        all_columns = [df.index.name] + list(df.columns)
        header_values = [x.replace(" ", "<br>") for x in all_columns]
        cells_values = [df.index.to_list()] + [df[x].to_list() for x in df.columns]

        fig.add_trace(
            go.Table(
                header=dict(values=header_values, font=dict(size=10), align="left"),
                cells=dict(values=cells_values, align="left"),
            ),
            row=1,
            col=1,
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

        for this_col in [
            gn.q_gas_m3day,
            gn.q_gas_m3day + gn.add_restore,
            gn.q_gas_m3day + gn.add_restore + gn.add_lr,
            gn.q_gas_m3day + gn.add_restore + gn.add_ridge,
            gn.q_gas_m3day + gn.add_restore + gn.add_lasso,
            gn.q_gas_m3day + gn.add_restore + gn.add_decision_tree,
            gn.q_gas_m3day + gn.add_restore + gn.add_random_forest,
        ]:
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

        for this_col in [
            gn.c_calibr_choke,
            gn.c_calibr_choke + gn.add_restore,
            gn.c_calibr_choke + gn.add_restore + gn.add_lr,
            gn.c_calibr_choke + gn.add_restore + gn.add_ridge,
            gn.c_calibr_choke + gn.add_restore + gn.add_lasso,
            gn.c_calibr_choke + gn.add_restore + gn.add_decision_tree,
            gn.c_calibr_choke + gn.add_restore + gn.add_random_forest,
        ]:
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
                    row=4,
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
                    row=5,
                    col=1,
                )

        fig.update_layout(
            title_text=f"Field={self.field}, Well={self.well_name}, Step=Predict",
            height=450 * 4,
        )

        fig.layout.hovermode = "x"

        plot(
            fig,
            filename=f"{self.settings.full_path_to_save}/{self.well_name}_after_prediction_view.html",
            auto_open=True,
        )
