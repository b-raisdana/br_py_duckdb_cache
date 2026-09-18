from pydantic_settings import BaseSettings

from config import Config


def test_config_exposes_log_path_and_environment(tmp_path):
    config = Config(root_path=tmp_path, environment="production")

    assert config.path_of_logs == tmp_path / "logs"
    assert config.environment == "production"
    assert isinstance(config, BaseSettings)
