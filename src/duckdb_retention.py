import pandas as pd

# mypy: ignore-errors


def remove_overlapping_ranges(indexes: pd.MultiIndex) -> pd.MultiIndex:
    all_tfs = indexes.get_level_values("timeframe").unique().sort_values(key=pd.to_timedelta, ascending=False)

    kept: list[pd.DataFrame] = []

    for tf in all_tfs:
        current = indexes[indexes.get_level_values("timeframe") == tf]
        dt = current.get_level_values("date")

        if kept:
            larger: pd.DataFrame = pd.concat(kept)
            larger_start = larger.get_level_values("date")
            larger_end = larger_start + pd.to_timedelta(larger.get_level_values("timeframe"))

            covered = pd.Series(False, index=dt)

            for start, end in zip(larger_start, larger_end, strict=False):
                covered |= (dt >= start) & (dt < end)

            current = current[~covered]

        if len(current):
            kept.append(current)

    return kept[0].append(kept[1:]) if kept else indexes[:0]
