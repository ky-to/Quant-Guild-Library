from .config import StrategyConfig
from .pipeline import main


if __name__ == "__main__":
    config = StrategyConfig(
        ticker="GOOGL",
        data_folder=r"D:\playground\Quant-Guild-Library\GOOGL_2018-05-01T09-30-00_to_2026-03-14T16-00-00",
        interval="1d",
        data_start="2015-01-01",
        data_end="",
        strategy_start="2023-01-01",
        train_bars=756,
        retrain_bars=126,
        min_hold_bars=1,
        confirmation_bars=1,
    )

    test_data, trades = main(config)