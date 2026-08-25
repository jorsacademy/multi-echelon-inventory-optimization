from __future__ import annotations

import pandas as pd


def summarize_history(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate period-level simulation history into node-level KPIs."""
    rows: list[dict[str, float | str]] = []

    for node, frame in history.groupby("Node", sort=False):
        demand_units = float(frame["Demand"].sum())
        filled_units = float(frame["Filled Demand"].sum())
        stockout_periods = int((frame["Backorders"] > 0).sum())
        periods = len(frame)

        rows.append(
            {
                "Node": node,
                "Avg On Hand": float(frame["On Hand"].mean()),
                "Avg Inventory Position": float(frame["Inventory Position"].mean()),
                "Avg Backorders": float(frame["Backorders"].mean()),
                "Demand Units": demand_units,
                "Filled Units": filled_units,
                "Fill Rate": 1.0 if demand_units == 0 else filled_units / demand_units,
                "Cycle Service Level": 1.0 - stockout_periods / periods,
                "Stockout Periods": stockout_periods,
                "Holding Cost": float(frame["Holding Cost"].sum()),
                "Shortage Cost": float(frame["Shortage Cost"].sum()),
                "Ordering Cost": float(frame["Ordering Cost"].sum()),
                "Total Cost": float(frame["Total Cost"].sum()),
            }
        )

    return pd.DataFrame(rows)
