from filtration import Filtration
from ml import ML
from reader import Reader
from solver import Solver


def get_virtual_flow_meter(data_load_type: str = "local"):
    """Метод запуска расчета ВР по модели Штуцера."""

    reader = Reader(data_load_type=data_load_type)

    filter = Filtration(
        data=reader.data,
        settings=reader.settings,
        q_column_to_use=reader.q_column_to_use,
        resample_time=reader.resample_time,
        field=reader.field,
        well=reader.well_name,
    )

    solver = Solver(
        data=filter.df_with_liq_rates,
        settings=reader.settings,
        q_column_to_use=reader.q_column_to_use,
        list_with_pvt_params=reader.list_with_pvt_params,
        field=reader.field,
        engine=reader.engine,
    )

    ml = ML(
        field=reader.field,
        well_name=reader.well_name,
        settings=reader.settings,
        df_with_liq_rate=solver.data,
        df_without_liq_rate=filter.df_without_liq_rates,
        q_data_type=reader.q_data_type,
        target=reader.target,
        predict=reader.predict,
        transform_type="standartization",
        q_column_to_use=reader.q_column_to_use,
    )

    solver.q_gas_calc(
        result_df=ml.df_with_predicted_calibr,
        target=reader.target,
        engine=reader.engine,
    )
    ml.restore_q_gas(q_gas_m3day_choke=solver.q_gas_m3day_choke)

    print(ml.error_qliq_df)


if __name__ == "__main__":
    get_virtual_flow_meter()
