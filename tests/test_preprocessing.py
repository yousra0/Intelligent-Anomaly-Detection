import pandas as pd

from src.preprocessing.data_loader import load_data


def test_load_data_reads_csv(tmp_path):
	csv_path = tmp_path / "sample.csv"
	source = pd.DataFrame(
		{
			"amount": [10.0, 20.5, 31.0],
			"isFraud": [0, 1, 0],
		}
	)
	source.to_csv(csv_path, index=False)

	loaded = load_data(csv_path, sample=False)

	assert loaded.shape == source.shape
	assert loaded.columns.tolist() == ["amount", "isFraud"]
	assert loaded["isFraud"].sum() == 1


def test_load_data_sampling_caps_to_dataset_size(tmp_path):
	csv_path = tmp_path / "sample_small.csv"
	source = pd.DataFrame({"x": [1, 2, 3], "isFraud": [0, 0, 1]})
	source.to_csv(csv_path, index=False)

	sampled = load_data(csv_path, sample=True, sample_size=100)

	assert len(sampled) == len(source)
	assert set(sampled.columns) == {"x", "isFraud"}


def test_load_data_rejects_unsupported_extension(tmp_path):
	invalid_path = tmp_path / "data.txt"
	invalid_path.write_text("a,b\n1,2\n", encoding="utf-8")

	try:
		load_data(invalid_path)
	except ValueError as exc:
		assert "Format non supporte" in str(exc) or "Format non supporté" in str(exc)
	else:
		raise AssertionError("load_data should reject unsupported file extensions")
