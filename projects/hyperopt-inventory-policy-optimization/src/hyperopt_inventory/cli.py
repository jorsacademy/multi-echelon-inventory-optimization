from .optimization import compare_searches


def main() -> None:
    for result in compare_searches(max_evals=20, seed=7):
        print(
            f"{result.algorithm}: s={result.reorder_point}, "
            f"S={result.order_up_to}, validation_cost={result.validation_loss:.3f}"
        )


if __name__ == "__main__":
    main()
