import pandas as pd

# mypy: ignore-errors


def remove_overlapping_ranges(indexes: pd.MultiIndex) -> pd.MultiIndex:
    all_tfs = indexes.get_level_values("timeframe").unique().sort_values(key=pd.to_timedelta, ascending=False)

    kept: list[pd.MultiIndex] = []

    for tf in all_tfs:
        current = indexes[indexes.get_level_values("timeframe") == tf]
        dt = current.get_level_values("date")

        if kept:
            larger: pd.MultiIndex = pd.MultiIndex.from_frame(
                pd.concat([item.to_frame(index=False) for item in kept], ignore_index=True)
            )
            larger_start = larger.get_level_values("date")
            larger_end = larger_start + pd.to_timedelta(larger.get_level_values("timeframe"))

            covered = pd.Series(False, index=dt)

            for start, end in zip(larger_start, larger_end, strict=False):
                covered |= (dt >= start) & (dt < end)

            current = current[~covered]

        if len(current):
            kept.append(current)

    if not kept:
        return indexes[:0]
    return pd.MultiIndex.from_frame(pd.concat([item.to_frame(index=False) for item in kept], ignore_index=True))
