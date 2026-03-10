import pandas as pd


def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract hour from step.
    In PaySim dataset: 1 step = 1 hour.
    """
    df["hour"] = df["step"] % 24
    return df


def add_day_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract day from step.
    """
    df["day"] = df["step"] // 24
    return df


def add_is_weekend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekend indicator.
    """
    if "day" not in df.columns:
        df["day"] = df["step"] // 24

    df["is_weekend"] = df["day"] % 7 >= 5
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all temporal feature transformations.
    """
    df = add_hour_feature(df)
    df = add_day_feature(df)
    df = add_is_weekend_feature(df)

    return df