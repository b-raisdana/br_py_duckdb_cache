from enum import auto

from config import app_config
from helper.enum_utils import AutoSnakeEnum


class DatastoreRegistry(AutoSnakeEnum):
    UnifiedNoNAN = auto()
    Tick = auto()

    @staticmethod
    def _namespace() -> str:
        return (
            f"{app_config.under_process_market}.{app_config.under_process_symbol}.{app_config.under_process_exchange}"  # noqa: F821
        )

    @staticmethod
    def _table_identifier(table_name: str) -> str:
        return f"{DatastoreRegistry._namespace()}.{table_name}"

    def get_auto_table_name(self) -> str:
        return DatastoreRegistry._namespace()
