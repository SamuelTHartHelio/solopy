"""Measure complete dropout intervals from an events CSV.

Run: python dropout_stats.py /path/to/events.csv --output-dir dropout_results
"""

import argparse
from itertools import groupby
from pathlib import Path

import pandas as pd


PAIR_COLUMNS = ["Event", "Type", "Dropout", "Dropin", "Duration_min"]
REVIEW_COLUMNS = ["Event", "Type", "Marker", "Time", "Reason"]


def extract_durations(events):
    """Pair one dropout with the next drop-in within each event window.

    If several dropouts precede the first drop-in, leave the whole set
    for review. Extra drop-ins cannot close a dropout twice. Events are
    processed independently, even when their windows overlap.
    """
    columns = ["Event", "Start", "Stop", "Type", "Dropout", "Dropin"]
    pairs, review = [], []
    for event, start, stop, event_type, dropouts, dropins in events[columns].itertuples(index=False, name=None):
        start, stop = pd.to_datetime(start, utc=True), pd.to_datetime(stop, utc=True)
        if not start < stop:
            raise ValueError(f"Event {event}: Start must be earlier than Stop")
        event_type = str(event_type).strip().casefold()
        markers = []
        for kind, cell in (("Dropout", dropouts), ("Dropin", dropins)):
            for value in str(cell).split(";") if pd.notna(cell) else ():
                if not value.strip():
                    continue
                time = pd.to_datetime(value.strip(), utc=True)
                if not start <= time <= stop:
                    review.append((event, event_type, kind, time, "outside event window"))
                else:
                    markers.append((time, kind))

        pending = []
        for time, group in groupby(sorted(markers), key=lambda marker: marker[0]):
            kinds = [kind for _, kind in group]
            if len(set(kinds)) > 1:
                review.extend((event, event_type, "Dropout", t, "drop-in ties with a dropout") for t in pending)
                review.extend((event, event_type, kind, time, "dropout and drop-in at same time") for kind in kinds)
                pending.clear()
            elif kinds[0] == "Dropout":
                pending.extend([time] * len(kinds))
            else:
                if len(pending) == 1:
                    duration = (time - pending[0]).total_seconds() / 60
                    pairs.append((event, event_type, pending[0], time, duration))
                elif pending:
                    reason = "multiple dropouts before one drop-in"
                    review.extend((event, event_type, "Dropout", t, reason) for t in pending)
                    review.append((event, event_type, "Dropin", time, reason))
                extra = len(kinds) - bool(pending)
                review.extend([(event, event_type, "Dropin", time, "unmatched drop-in")] * extra)
                pending.clear()

        review.extend((event, event_type, "Dropout", t, "no later in-window drop-in") for t in pending)
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS), pd.DataFrame(review, columns=REVIEW_COLUMNS)


def summarize(pairs):
    """Summarize complete durations in minutes overall and by event type."""
    rows = []
    for name, group in [("All", pairs), *pairs.groupby("Type")]:
        duration = group["Duration_min"]
        rows.append({
            "Type": name,
            "Pairs": len(group),
            "Events": group["Event"].nunique(),
            "Median_min": duration.median(),
            "Q1_min": duration.quantile(0.25),
            "Q3_min": duration.quantile(0.75),
            "Min_min": duration.min(),
            "Max_min": duration.max(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dropout_results"))
    args = parser.parse_args()

    events = pd.read_csv(args.events, dtype=str)
    pairs, review = extract_durations(events)
    summary = summarize(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "dropout_durations.csv", index=False)
    review.to_csv(args.output_dir / "dropout_review.csv", index=False)
    summary.to_csv(args.output_dir / "dropout_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(f"Review: {len(review)} markers across {review['Event'].nunique()} events.")


if __name__ == "__main__":
    main()
