import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from signal_pipeline import vol_signal_layer as vsl


def test_load_latest_score_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(vsl, "SIGNAL_DIR", str(tmp_path))
    assert vsl.load_latest_score("GME") is None


def test_load_latest_score_selects_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(vsl, "SIGNAL_DIR", str(tmp_path))
    data_old = pd.DataFrame([{"vol_container_score": 0.2}])
    data_new = pd.DataFrame([{"vol_container_score": 0.5}])
    old_file = tmp_path / "GME_20230101.csv"
    new_file = tmp_path / "GME_20240101.csv"
    data_old.to_csv(old_file, index=False)
    data_new.to_csv(new_file, index=False)
    result = vsl.load_latest_score("GME")
    assert result == data_new.iloc[-1].to_dict()


def test_evaluate_signal_high_alert():
    score_data = {"vol_container_score": 0.8}
    res = vsl.evaluate_signal(score_data)
    assert res["alerts"] == ["⚠️ High containment: short-vol setup ideal"]


def test_evaluate_signal_low_alert():
    score_data = {"vol_container_score": 0.2}
    res = vsl.evaluate_signal(score_data)
    assert res["alerts"] == ["📈 Breakdown risk: monitor for long-vol setup"]


def test_evaluate_signal_neutral_alert():
    score_data = {"vol_container_score": 0.5}
    res = vsl.evaluate_signal(score_data)
    assert res["alerts"] == ["📊 Neutral: hold or prepare"]


def test_evaluate_signal_no_data():
    res = vsl.evaluate_signal(None)
    assert res["alerts"] == ["No score data"]
