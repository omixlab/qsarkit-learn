"""Tests for pickle-free model persistence."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qsarkit.models import QSARClassifier, QSARRegressor
from qsarkit.persistence import (
    BUNDLE_FORMAT_VERSION,
    ModelBundle,
    ModelMetadata,
    environment_summary,
    inspect_bundle,
    load_model,
    save_model,
)
from qsarkit.representation import MorganFingerprint

skops = pytest.importorskip("skops.io")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.2, size=40)
    return X, y


@pytest.fixture(scope="module")
def labels(data):
    _, y = data
    return (y > np.median(y)).astype(int)


@pytest.fixture
def fitted(data):
    X, y = data
    return Ridge().fit(X, y)


class TestMetadata:
    def test_fills_in_provenance(self):
        meta = ModelMetadata(name="demo", endpoint="pIC50")
        assert meta.created
        assert meta.qsarkit_version
        assert "numpy" in meta.environment
        assert meta.format_version == BUNDLE_FORMAT_VERSION

    def test_rejects_an_unknown_task(self):
        with pytest.raises(ValueError, match="task must be"):
            ModelMetadata(task="clustering")

    def test_round_trips_through_a_dict(self):
        meta = ModelMetadata(name="a", endpoint="b", n_features=7)
        restored = ModelMetadata.from_dict(meta.to_dict())
        assert restored.name == "a"
        assert restored.n_features == 7

    def test_ignores_fields_a_newer_version_added(self):
        """A bundle from a future qsarkit must still load here."""
        payload = ModelMetadata(name="a").to_dict()
        payload["some_field_from_the_future"] = 42
        assert ModelMetadata.from_dict(payload).name == "a"

    def test_is_json_serializable(self):
        json.dumps(ModelMetadata(name="a").to_dict())

    def test_detects_no_change_in_the_same_session(self):
        assert ModelMetadata(name="a").environment_differences() == {}

    def test_detects_a_version_change(self):
        meta = ModelMetadata(name="a")
        meta.environment["numpy"] = "0.0.1"
        differences = meta.environment_differences()
        assert "numpy" in differences
        assert "0.0.1" in differences["numpy"]

    def test_environment_summary_names_the_core_packages(self):
        summary = environment_summary()
        assert {"qsarkit", "python", "numpy", "scikit-learn"} <= set(summary)


class TestSaveLoad:
    def test_round_trip_preserves_predictions(self, fitted, data, tmp_path):
        X, _ = data
        path = save_model(fitted, tmp_path / "m")
        assert np.allclose(load_model(path).predict(X), fitted.predict(X))

    def test_writes_the_documented_layout(self, fitted, tmp_path):
        path = Path(save_model(fitted, tmp_path / "m"))
        assert sorted(p.name for p in path.iterdir()) == [
            "estimator.skops",
            "manifest.json",
            "metadata.json",
        ]

    def test_metadata_and_manifest_are_plain_json(self, fitted, tmp_path):
        """Readable without importing qsarkit, which is the point."""
        path = Path(save_model(fitted, tmp_path / "m"))
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["format"] == "qsarkit-model-bundle"
        assert manifest["estimator_class"] == "sklearn.linear_model._ridge.Ridge"
        json.loads((path / "metadata.json").read_text())

    def test_appends_the_suffix_when_absent(self, fitted, tmp_path):
        assert save_model(fitted, tmp_path / "m").endswith(".qsar")

    def test_respects_an_explicit_suffix(self, fitted, tmp_path):
        assert save_model(fitted, tmp_path / "m.bundle").endswith(".bundle")

    def test_overwrites_by_default(self, fitted, tmp_path):
        save_model(fitted, tmp_path / "m")
        save_model(fitted, tmp_path / "m")

    def test_refuses_to_overwrite_when_told_not_to(self, fitted, tmp_path):
        save_model(fitted, tmp_path / "m")
        with pytest.raises(FileExistsError, match="overwrite"):
            save_model(fitted, tmp_path / "m", overwrite=False)

    def test_records_the_feature_count(self, fitted, tmp_path):
        assert load_model(save_model(fitted, tmp_path / "m")).metadata.n_features == 6

    def test_carries_a_pipeline(self, data, tmp_path):
        X, y = data
        pipeline = Pipeline([("scale", StandardScaler())]).fit(X)
        bundle = ModelBundle(Ridge().fit(pipeline.transform(X), y), pipeline=pipeline)
        reloaded = load_model(save_model(bundle, tmp_path / "m"))
        assert reloaded.pipeline is not None
        assert np.allclose(
            reloaded.predict(pipeline.transform(X)),
            bundle.predict(pipeline.transform(X)),
        )

    def test_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            load_model(tmp_path / "nothing")

    def test_a_directory_without_a_manifest_is_not_a_bundle(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="not a qsarkit"):
            load_model(tmp_path / "empty")

    def test_a_future_format_version_is_refused(self, fitted, tmp_path):
        path = Path(save_model(fitted, tmp_path / "m"))
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["format_version"] = "99"
        (path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="Upgrade qsarkit"):
            load_model(path)

    def test_a_missing_estimator_file_is_reported(self, fitted, tmp_path):
        path = Path(save_model(fitted, tmp_path / "m"))
        (path / "estimator.skops").unlink()
        with pytest.raises(ValueError, match="missing its"):
            load_model(path)


class TestQsarkitEstimators:
    def test_regressor_facade_round_trips(self, data, tmp_path):
        X, y = data
        model = QSARRegressor("rf", random_state=0).fit(X, y)
        reloaded = load_model(save_model(model, tmp_path / "m"))
        assert np.allclose(reloaded.predict(X), model.predict(X))

    def test_classifier_facade_round_trips(self, data, labels, tmp_path):
        X, _ = data
        model = QSARClassifier("rf", random_state=0).fit(X, labels)
        reloaded = load_model(
            save_model(
                model, tmp_path / "m",
                metadata=ModelMetadata(task="classification"),
            )
        )
        assert np.allclose(reloaded.predict_proba(X), model.predict_proba(X))

    def test_qsarkit_classes_need_no_explicit_trust(self, data, tmp_path):
        """Otherwise every user would have to list qsarkit's own classes."""
        X, y = data
        path = save_model(QSARRegressor("rf", random_state=0).fit(X, y), tmp_path / "m")
        load_model(path)   # must not raise

    def test_inspect_does_not_report_qsarkit_classes_as_untrusted(
        self, data, tmp_path
    ):
        X, y = data
        path = save_model(QSARRegressor("rf", random_state=0).fit(X, y), tmp_path / "m")
        assert inspect_bundle(path)["untrusted"] == []

    def test_molecule_pipeline_round_trips(self, tmp_path):
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in
                ("CCO", "CCN", "c1ccccc1", "CCCl", "CCC", "c1ccncc1")]
        fingerprint = MorganFingerprint(radius=2, n_bits=128)
        X = fingerprint.transform(mols)
        y = np.arange(len(mols), dtype=float)
        model = QSARRegressor("rf", random_state=0).fit(X, y)

        reloaded = load_model(
            save_model(model, tmp_path / "m", pipeline=fingerprint)
        )
        assert np.allclose(reloaded.predict_mols(mols), model.predict(X))


class TestGuardrails:
    def test_wrong_feature_width_is_refused(self, fitted, tmp_path):
        bundle = load_model(save_model(fitted, tmp_path / "m"))
        with pytest.raises(ValueError, match="expects 6 features but was given 3"):
            bundle.predict(np.zeros((2, 3)))

    def test_correct_width_passes(self, fitted, data, tmp_path):
        X, _ = data
        bundle = load_model(save_model(fitted, tmp_path / "m"))
        assert bundle.predict(X).shape == (len(X),)

    def test_predict_mols_without_a_pipeline_is_refused(self, fitted, tmp_path):
        bundle = load_model(save_model(fitted, tmp_path / "m"))
        with pytest.raises(ValueError, match="no pipeline"):
            bundle.predict_mols([])

    def test_predict_proba_on_a_regressor_is_refused(self, fitted, tmp_path):
        bundle = load_model(save_model(fitted, tmp_path / "m"))
        with pytest.raises(AttributeError, match="predict_proba"):
            bundle.predict_proba(np.zeros((2, 6)))

    def test_an_environment_change_warns(self, fitted, tmp_path):
        metadata = ModelMetadata(name="demo")
        metadata.environment["scikit-learn"] = "0.0.1"
        path = save_model(fitted, tmp_path / "m", metadata=metadata)
        with pytest.warns(UserWarning, match="scikit-learn"):
            load_model(path)

    def test_the_warning_can_be_silenced(self, fitted, tmp_path):
        metadata = ModelMetadata(name="demo")
        metadata.environment["scikit-learn"] = "0.0.1"
        path = save_model(fitted, tmp_path / "m", metadata=metadata)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_model(path, warn_on_environment_change=False)

    def test_repr_is_informative(self, fitted):
        text = repr(ModelBundle(fitted, metadata=ModelMetadata(name="demo")))
        assert "demo" in text and "regression" in text


class TestPickleFallback:
    """The joblib escape hatch.

    Note what is *not* tested here: an object that skops rejects but pickle
    accepts. In practice skops falls back to pickling unknown attributes
    itself, so a dump only fails when pickling would fail too -- a thread
    lock, a generator, an open socket -- and then the fallback cannot help
    either. The fallback exists for skops' own unsupported cases (exotic
    dtypes, future versions tightening what they accept), so these tests
    simulate that failure rather than contriving an object class that does
    not really exist.
    """

    @staticmethod
    def _break_skops(monkeypatch, message="simulated skops failure"):
        import skops.io as skops_io

        def _fail(*args, **kwargs):
            raise TypeError(message)

        monkeypatch.setattr(skops_io, "dump", _fail)

    def test_a_skops_failure_is_refused_by_default(
        self, fitted, tmp_path, monkeypatch
    ):
        self._break_skops(monkeypatch)
        with pytest.raises(TypeError, match="allow_pickle_fallback"):
            save_model(fitted, tmp_path / "m")

    def test_the_error_explains_the_alternative(
        self, fitted, tmp_path, monkeypatch
    ):
        self._break_skops(monkeypatch)
        with pytest.raises(TypeError) as caught:
            save_model(fitted, tmp_path / "m")
        message = str(caught.value)
        assert "scikit-learn" in message
        assert "unsafe to load" in message

    def test_the_fallback_works_and_warns(
        self, fitted, data, tmp_path, monkeypatch
    ):
        X, _ = data
        self._break_skops(monkeypatch)
        with pytest.warns(UserWarning, match="pickle"):
            path = save_model(
                fitted, tmp_path / "m", allow_pickle_fallback=True
            )
        assert (Path(path) / "estimator.joblib").is_file()
        assert np.allclose(load_model(path).predict(X), fitted.predict(X))

    def test_the_manifest_records_which_format_was_used(
        self, fitted, tmp_path, monkeypatch
    ):
        self._break_skops(monkeypatch)
        with pytest.warns(UserWarning):
            path = Path(
                save_model(fitted, tmp_path / "m", allow_pickle_fallback=True)
            )
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["files"]["estimator"] == "estimator.joblib"

    def test_the_warning_names_the_consequence(
        self, fitted, tmp_path, monkeypatch
    ):
        """A user reaching for this must be told what they are accepting."""
        self._break_skops(monkeypatch)
        with pytest.warns(UserWarning) as caught:
            save_model(fitted, tmp_path / "m", allow_pickle_fallback=True)
        message = str(caught[0].message)
        assert "executes code" in message
        assert "different scikit-learn" in message


class TestInspect:
    def test_reports_the_manifest_and_metadata(self, fitted, tmp_path):
        path = save_model(
            fitted, tmp_path / "m", metadata=ModelMetadata(name="demo", endpoint="e")
        )
        report = inspect_bundle(path)
        assert report["manifest"]["format"] == "qsarkit-model-bundle"
        assert report["metadata"]["endpoint"] == "e"

    def test_reconstructs_nothing(self, fitted, tmp_path, monkeypatch):
        """Inspecting a stranger's bundle must not execute its contents."""
        path = save_model(fitted, tmp_path / "m")

        def _explode(*args, **kwargs):
            raise AssertionError("inspect_bundle must not load objects")

        monkeypatch.setattr(skops, "load", _explode)
        inspect_bundle(path)

    def test_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            inspect_bundle(tmp_path / "nothing")

    def test_a_directory_without_a_manifest_is_reported(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="manifest"):
            inspect_bundle(tmp_path / "empty")
