import pandas as pd


def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour from step.
    In PaySim dataset: 1 step = 1 hour.
    """
    out = df.copy()
    out["hour"] = out["step"] % 24
    return out


def add_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract day from step.
    """
    out = df.copy()
    out["day"] = out["step"] // 24
    return out


def add_week_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract week index from step.
    In PaySim dataset: 1 week = 168 steps.
    """
    out = df.copy()
    out["week"] = out["step"] // 168
    return out


def add_is_weekend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekend indicator.
    """
    out = df.copy()
    if "day" not in out.columns:
        out["day"] = out["step"] // 24

    out["is_weekend"] = (out["day"] % 7 >= 5).astype(int)
    return out


def create_temporal_features(
    df: pd.DataFrame,
    include_is_weekend: bool = False,
) -> pd.DataFrame:
    """
    Apply all temporal feature transformations.
    """
    out = add_hour_feature(df)
    out = add_day_feature(out)
    out = add_week_feature(out)
    if include_is_weekend:
        out = add_is_weekend_feature(out)
    return out