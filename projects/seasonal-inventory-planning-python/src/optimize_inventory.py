from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def deterministic_replenishment_cost(
    order_quantity: int,
    demand: int,
    ordering_cost: float,
    holding_cost: float,
) -> float:
    """Return relevant replenishment cost for an integer order quantity.

    Unit purchase cost is intentionally excluded because it is constant with
    respect to order quantity when all seasonal demand must be purchased at a
    fixed unit cost.
    """
    if order_quantity <= 0:
        raise ValueError("order_quantity must be positive")

    number_of_orders = int(np.ceil(demand / order_quantity))
    ordering_component = number_of_orders * ordering_cost

    # Average cycle inventory is approximated as Q / 2. Multiplying by the
    # number of cycles would double-count holding cost when holding_cost is
    # specified on a season-wide per-unit basis, so the seasonal holding term
    # is applied once to average inventory.
    holding_component = holding_cost * (order_quantity / 2.0)

    return ordering_component + holding_component


def find_best_integer_order_quantity(
    demand: int,
    ordering_cost: float,
    holding_cost: float,
) -> tuple[int, float]:
    """Enumerate feasible integer quantities and return the minimum-cost choice."""
    quantities = np.arange(1, demand + 1, dtype=int)
    costs = np.array(
        [
            deterministic_replenishment_cost(q, demand, ordering_cost, holding_cost)
            for q in quantities
        ]
    )
    best_index = int(np.argmin(costs))
    return int(quantities[best_index]), float(costs[best_index])


def analyze_inventory(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate EOQ, integer replenishment, reorder point, and newsvendor results."""
    rows = []

    for _, item in data.iterrows():
        demand = int(item["Forecast_Demand"])
        demand_std = float(item["Demand_Std"])
        ordering_cost = float(item["Ordering_Cost"])
        holding_cost = float(item["Holding_Cost"])
        unit_cost = float(item["Unit_Cost"])
        selling_price = float(item["Selling_Price"])
        salvage_value = float(item["Salvage_Value"])
        season_length = int(item["Season_Length_Weeks"])
        lead_time = int(item["Lead_Time_Weeks"])
        service_level = float(item["Target_Service_Level"])

        eoq = np.sqrt((2.0 * ordering_cost * demand) / holding_cost)
        best_q, relevant_cost = find_best_integer_order_quantity(
            demand=demand,
            ordering_cost=ordering_cost,
            holding_cost=holding_cost,
        )

        weekly_mean = demand / season_length
        weekly_std = demand_std / np.sqrt(season_length)
        z_service = norm.ppf(service_level)

        expected_lead_time_demand = weekly_mean * lead_time
        safety_stock = z_service * weekly_std * np.sqrt(lead_time)
        reorder_point = expected_lead_time_demand + safety_stock

        underage_cost = selling_price - unit_cost
        overage_cost = unit_cost - salvage_value
        critical_fractile = underage_cost / (underage_cost + overage_cost)
        critical_fractile = float(np.clip(critical_fractile, 1e-6, 1 - 1e-6))
        newsvendor_z = norm.ppf(critical_fractile)
        newsvendor_quantity = demand + newsvendor_z * demand_std

        rows.append(
            {
                "Item_ID": item["Item_ID"],
                "Forecast_Demand": demand,
                "EOQ_Benchmark": round(float(eoq), 2),
                "Optimal_Integer_Order_Qty": best_q,
                "Estimated_Relevant_Cost": round(relevant_cost, 2),
                "Safety_Stock": round(float(safety_stock), 2),
                "Reorder_Point": int(np.ceil(reorder_point)),
                "Critical_Fractile": round(critical_fractile, 4),
                "Newsvendor_Order_Qty": max(0, int(np.ceil(newsvendor_quantity))),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "seasonal_inventory_data.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run src/generate_data.py first."
        )

    data = pd.read_csv(data_path)
    results = analyze_inventory(data)

    output_path = results_dir / "inventory_optimization_results.csv"
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
