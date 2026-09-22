import numpy as np
import torch

from contextual_opt.cost import newsvendor_cost_numpy
from contextual_opt.data import generate_contextual_demand
from contextual_opt.policies import (
    DirectDecisionPolicy,
    LinearQuantilePolicy,
    knn_saa,
    train_direct,
    train_quantile,
)


def run(seed: int) -> dict[str, float]:
    features, demand = generate_contextual_demand(800, seed=seed)
    train_x, train_y = features[:500], demand[:500]
    test_x, test_y = features[500:], demand[500:]
    train_x_t = torch.tensor(train_x)
    train_y_t = torch.tensor(train_y)

    torch.manual_seed(seed)
    quantile_model = train_quantile(
        LinearQuantilePolicy(train_x.shape[1]),
        train_x_t,
        train_y_t,
    )
    torch.manual_seed(seed)
    direct_model = train_direct(
        DirectDecisionPolicy(train_x.shape[1]),
        train_x_t,
        train_y_t,
    )

    with torch.no_grad():
        pto_quantity = quantile_model(torch.tensor(test_x)).numpy()
        direct_quantity = direct_model(torch.tensor(test_x)).numpy()
    local_quantity = knn_saa(train_x, train_y, test_x, k=40)

    return {
        "predict_then_optimize": float(newsvendor_cost_numpy(pto_quantity, test_y).mean()),
        "direct_decision": float(newsvendor_cost_numpy(direct_quantity, test_y).mean()),
        "knn_saa": float(newsvendor_cost_numpy(local_quantity, test_y).mean()),
    }


def main() -> None:
    results = [run(seed) for seed in range(5)]
    for name in results[0]:
        values = np.asarray([result[name] for result in results])
        print(
            name,
            f"mean_cost={values.mean():.4f}",
            f"std={values.std(ddof=1):.4f}",
        )


if __name__ == "__main__":
    main()
