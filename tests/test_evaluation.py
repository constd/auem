import string

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

import auem.evaluation as au_eval

np.random.seed(12345)


def generate_example(n_classes, p_match=0.5, n_sources=3):
    target = np.random.randint(0, n_classes)
    if np.random.random() < p_match:
        y_pred = target
    else:
        y_pred = np.random.randint(0, n_classes)

    y_score = softmax(np.random.random(n_classes))
    # shift it so the max is the same as y_pred'
    y_score = np.roll(y_score, y_pred - np.argmax(y_score))

    assert np.argmax(y_score) == y_pred

    source = np.random.choice(list(string.ascii_letters[:n_sources]))

    return {"y_true": target, "y_pred": y_pred, "y_score": y_score, "source": source}


@pytest.fixture
def n_examples():
    return 20


@pytest.fixture(params=[2, 10])
def n_classes(request):
    return request.param


@pytest.fixture(params=[1, 6])
def n_sources(request):
    return request.param


class TestPredictionReport:
    @pytest.fixture
    def example_preds_df(self, request, n_examples, n_sources, n_classes):
        df = pd.DataFrame(
            [
                generate_example(n_classes, n_sources=n_sources)
                for _ in range(n_examples)
            ]
        )
        assert len(df) == n_examples
        return df

    def test_prediction_report_df(self, example_preds_df, n_examples, n_sources):
        result_df = au_eval.prediction_report_df(
            example_preds_df, with_formatting=False
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == (len(example_preds_df["y_true"].unique()) + 1)
        # Only two columns without the source specified - accuracy and logloss
        assert len(result_df.columns) == 2

        assert result_df.iloc[-1].name == "Average"
        assert result_df.columns[-1] == "Log Loss"

    def test_prediction_report_df__with_formatting(
        self, example_preds_df, n_examples, n_sources
    ):
        result_df = au_eval.prediction_report_df(example_preds_df, with_formatting=True)
        assert isinstance(result_df, pd.io.formats.style.Styler)
        assert isinstance(result_df._repr_html_(), str)

    def test_prediction_report_df__with_source(
        self, example_preds_df, n_examples, n_sources, n_classes
    ):
        result_df = au_eval.prediction_report_df(
            example_preds_df, source_column="source", with_formatting=False
        )
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == (len(example_preds_df["y_true"].unique()) + 1)
        assert len(result_df.columns) == (len(example_preds_df["source"].unique()) + 2)

        assert result_df.iloc[-1].name == "Average"
        assert result_df.columns[-1] == "Log Loss"
