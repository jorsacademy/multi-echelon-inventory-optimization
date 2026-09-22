from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_ITEMS = 10
SEASON_LENGTH_WEEKS = 12


def generate_inventory_data(
    num_items: int = NUM_ITEMS,
    season_length_weeks: int = SEASON_LENGTH_WEEKS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset for seasonal inventory planning."""
    rng = np.random.default_rng(random_seed)

    forecast_demand = rng.integers(80, 500, size=num_items)
    demand_cv = rng.uniform(0.12, 0.30, size=num_items)
    demand_std = np.maximum(5.0, forecast_demand * demand_cv)

    unit_cost = rng.uniform(10.0, 30.0, size=num_items)
    gross_margin = rng.uniform(0.30, 0.65, size=num_items)
    selling_price = unit_cost / (1.0 - gross_margin)

    salvage_fraction = rng.uniform(0.10, 0.45, size=num_items)
    salvage_value = unit_cost * salvage_fraction

    ordering_cost = rng.uniform(25.0, 90.0, size=num_items)
    holding_cost = rng.uniform(0.60, 2.50, size=num_items)
    lead_time_weeks = rng.integers(1, 4, size=num_items)
    target_service_level = rng.choice(
        [0.90, 0.95, 0.975],
        size=num_items,
        p=[0.25, 0.50, 0.25],
    )

    data = pd.DataFrame(
        {
            "Item_ID": [f"Item_{i + 1}" for i in range(num_items)],
            "Forecast_Demand": forecast_demand.astype(int),
            "Demand_Std": np.round(demand_std, 2),
            "Ordering_Cost": np.round(ordering_cost, 2),
            "Holding_Cost": np.round(holding_cost, 2),
            "Unit_Cost": np.round(unit_cost, 2),
            "Selling_Price": np.round(selling_price, 2),
            "Salvage_Value": np.round(salvage_value, 2),
            "Season_Length_Weeks": season_length_weeks,
            "Lead_Time_Weeks": lead_time_weeks.astype(int),
            "Target_Service_Level": target_service_level,
        }
    )

    return data


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "seasonal_inventory_data.csv"
    data = generate_inventory_data()
    data.to_csv(output_path, index=False)

    print(f"Generated {len(data)} synthetic items.")
    print(f"Saved dataset to: {output_path}")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
