from multi_echelon_inventory import (
    InventoryNode,
    MultiEchelonSystem,
    PolicyType,
    SimulationConfig,
    compare_policies,
)


def main() -> None:
    nodes = [
        InventoryNode(
            name="Supplier",
            initial_on_hand=1000,
            demand_mean=20,
            demand_std=6,
            lead_time=10,
            lead_time_std=2.0,
            holding_cost=0.10,
            shortage_cost=100,
            ordering_cost=25,
            service_level=0.95,
            shipment_capacity=80,
            production_capacity=70,
            order_capacity=100,
        ),
        InventoryNode(
            name="Manufacturer",
            initial_on_hand=750,
            demand_mean=20,
            demand_std=6,
            lead_time=5,
            lead_time_std=1.0,
            holding_cost=0.20,
            shortage_cost=200,
            ordering_cost=20,
            service_level=0.95,
            shipment_capacity=60,
            production_capacity=55,
            order_capacity=80,
        ),
        InventoryNode(
            name="Distributor",
            initial_on_hand=500,
            demand_mean=20,
            demand_std=6,
            lead_time=3,
            lead_time_std=0.75,
            holding_cost=0.15,
            shortage_cost=150,
            ordering_cost=15,
            service_level=0.95,
            shipment_capacity=50,
            order_capacity=60,
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

    print("\nSingle-run simulation summary:")
    print(result.summary.to_string(index=False))

    comparison = compare_policies(
        system,
        runs=100,
        periods=365,
        base_seed=42,
        policies=[
            PolicyType.BASE_STOCK,
            PolicyType.CRITICAL_RATIO,
            PolicyType.ECHELON_BASE_STOCK,
        ],
    )

    print("\nMonte Carlo policy comparison:")
    print(comparison.policy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
