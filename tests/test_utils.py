import numpy as np
import pandas as pd

from src.utils.anomaly_utils import check_missing_values


def test_check_missing_values_counts_and_percentages():
	df = pd.DataFrame(
		{
			"a": [1.0, np.nan, 3.0, np.nan],
			"b": [10, 11, 12, 13],
			"c": [np.nan, np.nan, 5.0, 6.0],
		}
	)

	missing = check_missing_values(df)

	assert missing.loc["a", "Missing Values"] == 2
	assert missing.loc["c", "Missing Values"] == 2
	assert missing.loc["b", "Missing Values"] == 0
	assert missing.loc["a", "Percentage"] == 50.0
	assert missing.loc["b", "Percentage"] == 0.0


def test_check_missing_values_sorted_by_percentage_desc():
	df = pd.DataFrame(
		{
			"x": [1, np.nan, np.nan],
			"y": [1, 2, 3],
			"z": [np.nan, 2, 3],
		}
	)

	missing = check_missing_values(df)
	ordered_cols = missing.index.tolist()

	assert ordered_cols[0] == "x"
	assert ordered_cols[-1] == "y"
