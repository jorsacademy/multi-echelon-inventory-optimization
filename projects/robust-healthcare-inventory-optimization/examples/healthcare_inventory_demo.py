from robust_inventory import RobustInventoryOptimizer


def main() -> None:
    optimizer = RobustInventoryOptimizer(num_scenarios=250, seed=42)
    policy = optimizer.solve_heuristic()
    analysis = optimizer.analyze(policy)

    print("Robust Healthcare Inventory Optimization")
    print(f"Initial units: {analysis['initial_units']:,}")
    print(f"Initial investment: ${analysis['initial_investment']:,.2f}")
    print(f"Expected operating cost: ${analysis['expected_total_cost']:,.2f}")
    print(f"CVaR(95%): ${analysis['cvar_95']:,.2f}")
    print("Service levels:")
    for segment, value in analysis["service_levels"].items():
        print(f"  {segment[0]} / {segment[1]}: {value:.2%}")


if __name__ == "__main__":
    main()
