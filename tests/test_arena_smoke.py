from forecast_arena import Protocol, load_records, per_task_table, run_arena


def test_arena_runs_resumes_and_ranks(tmp_path):
    models = ["Naive", "Drift", "Mean", "Ridge"]
    protocol = Protocol(n_origins=3)

    df = run_arena(["henon-x"], models, protocol=protocol, output_dir=tmp_path, verbose=False)
    assert len(df) == len(models) * 3
    assert df["error"].isna().all()
    assert (df["n_test"] == df["n_test"].iloc[0]).all()
    assert df["data_sha256"].nunique() == 1

    # A second call with resume finds nothing to do and returns the same records.
    again = run_arena(["henon-x"], models, protocol=protocol, output_dir=tmp_path, verbose=False)
    assert len(again) == len(df)
    assert len(load_records(tmp_path)) == len(df)

    table = per_task_table(df)
    assert set(table["model"]) == set(models)
    assert table["rank"].min() == 1
    # Over a long horizon on a bounded chaotic map a linear trend diverges, so Drift ranks last.
    assert table.sort_values("rank").iloc[-1]["model"] == "Drift"
