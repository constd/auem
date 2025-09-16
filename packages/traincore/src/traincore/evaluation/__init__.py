from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def _highlight_max(s):
    """
    highlight the maximum in a Series yellow.
    """
    is_max = s == s.max()
    return ["background-color: yellow" if v else "" for v in is_max]


def _highlight_min(s):
    """
    highlight the maximum in a Series yellow.
    """
    is_min = s == s.min()
    return ["background-color: red" if v else "" for v in is_min]


def _highlight_gray(s):
    """
    highlight the maximum in a Series yellow.
    """
    return [
        "background-color: #ccc" if i == (len(s) - 1) else "" for i, v in enumerate(s)
    ]


def df_logloss(group_df: pd.DataFrame, labels: List) -> pd.Series:
    """log loss for a group"""
    scores = []
    for idx, row in group_df.iterrows():
        scores.append([float(x) for x in row["y_score"]])
    loss = log_loss(group_df["y_true"], scores, labels=labels)
    return pd.Series([loss], index=["Log Loss"])


def prediction_report_df(
    preds_df: pd.DataFrame,
    target_column: str = "y_true",
    predictions_column: str = "y_pred",
    score_column: str = "y_score",
    source_column: Optional[str] = None,
    primary_metric: str = "accuracy",
    secondary_metrics: List[str] = ["logloss"],
    target_names: Dict[Any, str] = None,
    with_formatting: bool = True,
) -> pd.DataFrame:
    """Create a "report dataframe", given a dataset's targets and predictions.

    Will optionally break the results down by "source", if a source_column is provided.
    Reports results by "macro" average.
    """
    preds_df_copy = preds_df.copy(deep=False)

    if primary_metric == "accuracy":
        preds_df_copy["accuracy"] = (
            preds_df_copy["y_pred"] == preds_df_copy["y_true"]
        ).astype(float)

    else:
        raise NotImplementedError()

    if target_names is None:
        max_target = max(preds_df_copy["y_true"].unique())
        target_names = [str(x) for x in range(0, max_target + 1)]

    preds_df_copy["label"] = preds_df_copy["y_true"].apply(lambda x: target_names[x])

    scene_accuracy = preds_df_copy.groupby("label").agg(np.mean)["accuracy"]
    scene_logLoss = preds_df_copy.groupby("label").apply(
        partial(df_logloss, labels=target_names)
    )

    if source_column is not None:
        pivot_summary = pd.pivot_table(
            preds_df_copy,
            values="accuracy",
            index=["label"],
            columns=[source_column],
            aggfunc=np.mean,
        )
        final_df = pd.concat([scene_accuracy, pivot_summary, scene_logLoss], axis=1)
    else:
        final_df = pd.concat([scene_accuracy, scene_logLoss], axis=1)

    # add a summary column at the bottom.
    final_df.loc["Average"] = final_df.mean()

    if with_formatting:
        final_df_pct = final_df
        final_df_pct[final_df.columns[:-1]] *= 100

        styled_df = final_df_pct.style.apply(_highlight_max).apply(_highlight_min)
        return styled_df.apply(_highlight_gray)
    else:
        return final_df
