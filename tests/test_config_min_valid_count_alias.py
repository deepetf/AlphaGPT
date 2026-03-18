import os
import sys
import tempfile
import textwrap
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_core.config_loader import load_config, reset_config


def test_legacy_min_valid_count_maps_to_signal_min_valid_count():
    reset_config()
    yaml_text = textwrap.dedent(
        """
        robust_config:
          min_valid_count: 17
        """
    ).strip()

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        handle.write(yaml_text)
        temp_path = handle.name

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = load_config(temp_path)
        assert config["robust_config"]["signal_min_valid_count"] == 17
        assert config["robust_config"]["min_valid_count"] == 17
        assert any("min_valid_count 已弃用" in str(item.message) for item in caught)
    finally:
        reset_config()
        os.unlink(temp_path)
