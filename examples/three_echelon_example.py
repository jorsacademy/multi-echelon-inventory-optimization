from multi_echelon_inventory import (
    InventoryNode,
    MultiEchelonSystem,
    PolicyType,
    SimulationConfig,
)


def main() -> None:
    nodes = [
        InventoryNode(
            name="Supplier",
            initial_on_hand=1000,
            demand_mean=50,
            demand_std=10,
            lead_time=10,
            holding_cost=0.10,
            shortage_cost=100,
            ordering_cost=25,
            service_level=0.95,
        ),
        InventoryNode(
            name="Manufacturer",
            initial_on_hand=750,
            demand_mean=30,
            demand_std=8,
            lead_time=5,
            holding_cost=0.20,
            shortage_cost=200,
            ordering_cost=20,
            service_level=0.95,
        ),
        InventoryNode(
            name="Distributor",
            initial_on_hand=500,
            demand_mean=20,
            demand_std=6,
            lead_time=3,
            holding_cost=0.15,
            shortage_cost=150,
            ordering_cost=15,
            service_level=0.95,
        ),
    ]

    system = MultiEchelonSystem(nodes)

    print("Recommended echelon base-stock targets:")
    for name, target in system.recommended_stock_levels(
        PolicyType.ECHELON_BASE_STOCK
    ).items():
        print(f"  {name}: {target:.2f}")

    result = system.simulate(
        SimulationConfig(periods=365, seed=42),
        policy=PolicyType.ECHELON_BASE_STOCK,
    )

    print("\nSimulation summary:")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
