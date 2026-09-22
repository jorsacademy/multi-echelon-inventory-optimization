from __future__ import annotations

import argparse
import numpy as np

from conformal_inventory import (
    ClassicalSigmaBox,
    DemandPredictor,
    InventoryParameters,
    MarginalResidualBox,
    SimultaneousConformalBox,
    evaluate_methods,
    evaluate_true_distribution_reference,
    generate_dataset,
    marginal_coverage,
    residual_scale_from_training,
    rmse,
    simultaneous_coverage,
)


def run_experiment(args):
    train = generate_dataset(args.train_samples, seed=args.seed, horizon=args.horizon)
    calibration = generate_dataset(
        args.calibration_samples,
        seed=args.seed + 1_000_000,
        horizon=args.horizon,
    )
    test = generate_dataset(
        args.test_samples,
        seed=args.seed + 2_000_000,
        horizon=args.horizon,
    )

    if args.predictor == "extra-trees":
        predictor = DemandPredictor.fit(
            train.features,
            train.demand,
            seed=args.seed,
            n_estimators=args.trees,
        )
    else:
        try:
            from conformal_inventory.bnn import BayesianDemandPredictor
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise SystemExit(
                    "BNN mode requires PyTorch. Install with: "
                    "pip install -r requirements-bnn.txt"
                ) from exc
            raise
        predictor = BayesianDemandPredictor.fit(
            train.features,
            train.demand,
            seed=args.seed,
            hidden_dims=(args.bnn_hidden, args.bnn_hidden),
            prior_sigma=args.bnn_prior_sigma,
            noise_std=args.bnn_noise_std,
            learning_rate=args.bnn_learning_rate,
            steps=args.bnn_steps,
            train_samples=args.bnn_train_samples,
            prediction_samples=args.bnn_prediction_samples,
        )
    pred_train = predictor.predict(train.features)
    pred_cal = predictor.predict(calibration.features)
    pred_test = predictor.predict(test.features)

    scale = residual_scale_from_training(train.demand, pred_train)
    conformal = SimultaneousConformalBox.calibrate(
        calibration.demand,
        pred_cal,
        alpha=args.alpha,
        scale=scale,
    )
    marginal = MarginalResidualBox.calibrate(
        calibration.demand,
        pred_cal,
        alpha=args.alpha,
    )
    sigma = ClassicalSigmaBox.fit(
        train.demand,
        pred_train,
        sigma_multiplier=args.sigma_multiplier,
    )

    conf_lo, conf_hi = conformal.bounds(pred_test)
    marg_lo, marg_hi = marginal.bounds(pred_test)
    sig_lo, sig_hi = sigma.bounds(pred_test)

    print("=" * 104)
    print("SIMULTANEOUS CONFORMAL DEMAND BANDS")
    print(f"predictor                          : {args.predictor}")
    print("=" * 104)
    print(f"train RMSE                         : {rmse(train.demand, pred_train):8.3f}")
    print(f"calibration RMSE                   : {rmse(calibration.demand, pred_cal):8.3f}")
    print(f"test RMSE                          : {rmse(test.demand, pred_test):8.3f}")
    print(f"target trajectory coverage         : {1.0-args.alpha:8.3f}")
    print(f"conformal trajectory coverage      : {simultaneous_coverage(test.demand, conf_lo, conf_hi):8.3f}")
    print(f"marginal-box trajectory coverage   : {simultaneous_coverage(test.demand, marg_lo, marg_hi):8.3f}")
    print(f"2-sigma trajectory coverage        : {simultaneous_coverage(test.demand, sig_lo, sig_hi):8.3f}")
    print(f"conformal pointwise coverage       : {marginal_coverage(test.demand, conf_lo, conf_hi):8.3f}")
    print(f"marginal-box pointwise coverage    : {marginal_coverage(test.demand, marg_lo, marg_hi):8.3f}")
    print(f"mean conformal band width          : {np.mean(conf_hi-conf_lo):8.3f}")
    print(f"mean marginal band width           : {np.mean(marg_hi-marg_lo):8.3f}")
    print(f"mean sigma band width              : {np.mean(sig_hi-sig_lo):8.3f}")
    print(f"conformal normalized radius        : {conformal.radius:8.3f}")
    if args.predictor == "bnn":
        epistemic = predictor.epistemic_std(
            test.features,
            num_samples=args.bnn_prediction_samples,
        )
        print(f"mean BNN epistemic std             : {np.mean(epistemic):8.3f}")

    n_plan = min(args.planning_samples, args.test_samples)
    params = InventoryParameters.default(args.horizon)
    methods = evaluate_methods(
        pred_test[:n_plan],
        test.demand[:n_plan],
        params,
        interval_builders={
            "Classical sigma robust": sigma.bounds,
            "Marginal quantile robust": marginal.bounds,
            "Conformal robust": conformal.bounds,
        },
    )

    print()
    print("=" * 104)
    print(f"OUT-OF-SAMPLE INVENTORY DECISIONS ({n_plan} held-out contexts)")
    print("=" * 104)
    print(
        f"{'method':<27}{'mean cost':>12}{'fill rate':>12}"
        f"{'stockout':>12}{'clairvoyant gap':>18}{'order qty':>12}"
    )
    for row in methods:
        print(
            f"{row.name:<27}{row.mean_cost:12.2f}{row.mean_fill_rate:12.4f}"
            f"{row.mean_stockout_rate:12.4f}{row.mean_clairvoyant_gap_pct:17.2f}%"
            f"{row.mean_order_quantity:12.2f}"
        )

    if args.distribution_reference_contexts > 0:
        n_ref = min(args.distribution_reference_contexts, args.test_samples)
        ref = evaluate_true_distribution_reference(
            predictor,
            test.features[:n_ref],
            params,
            interval_builders={
                "Classical sigma robust": sigma.bounds,
                "Marginal quantile robust": marginal.bounds,
                "Conformal robust": conformal.bounds,
            },
            seed=args.seed + 7_000_000,
            saa_scenarios=args.saa_scenarios,
            evaluation_scenarios=args.reference_eval_scenarios,
        )
        print()
        print("=" * 104)
        print(
            f"TRUE-DISTRIBUTION REFERENCE ({n_ref} contexts; "
            f"information-advantaged SAA)"
        )
        print("=" * 104)
        print(
            f"{'method':<27}{'MC mean cost':>15}{'fill rate':>12}"
            f"{'stockout':>12}{'order qty':>12}"
        )
        for row in ref:
            print(
                f"{row.name:<27}{row.mean_cost:15.2f}{row.mean_fill_rate:12.4f}"
                f"{row.mean_stockout_rate:12.4f}{row.mean_order_quantity:12.2f}"
            )
        print(
            "The distribution-aware SAA reference can sample the true synthetic "
            "conditional demand generator. It is an information-advantaged "
            "reference, not a deployable baseline or an exact lower bound."
        )

    print()
    print(
        "Coverage is a statistical property of the demand band. "
        "Fill rate and stockout rate are downstream operational metrics; "
        "they are intentionally reported separately."
    )


def self_test():
    data = generate_dataset(20, seed=7, horizon=4)
    predictor = DemandPredictor.fit(
        data.features[:12],
        data.demand[:12],
        seed=7,
        n_estimators=40,
    )
    pred_train = predictor.predict(data.features[:12])
    pred_cal = predictor.predict(data.features[12:16])
    scale = residual_scale_from_training(data.demand[:12], pred_train)
    box = SimultaneousConformalBox.calibrate(
        data.demand[12:16],
        pred_cal,
        alpha=0.2,
        scale=scale,
    )
    lo, hi = box.bounds(predictor.predict(data.features[16:]))
    assert np.all(lo >= 0)
    assert np.all(hi >= lo)
    print("Conformal robust inventory self-test: OK")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--train-samples", type=int, default=700)
    p.add_argument("--calibration-samples", type=int, default=250)
    p.add_argument("--test-samples", type=int, default=700)
    p.add_argument("--planning-samples", type=int, default=80)
    p.add_argument(
        "--predictor",
        choices=("extra-trees", "bnn"),
        default="extra-trees",
        help="Point predictor used before conformal calibration.",
    )
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--bnn-hidden", type=int, default=32)
    p.add_argument("--bnn-steps", type=int, default=600)
    p.add_argument("--bnn-learning-rate", type=float, default=0.01)
    p.add_argument("--bnn-prior-sigma", type=float, default=1.0)
    p.add_argument("--bnn-noise-std", type=float, default=0.35)
    p.add_argument("--bnn-train-samples", type=int, default=3)
    p.add_argument("--bnn-prediction-samples", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--sigma-multiplier", type=float, default=2.0)
    p.add_argument("--distribution-reference-contexts", type=int, default=5)
    p.add_argument("--saa-scenarios", type=int, default=192)
    p.add_argument("--reference-eval-scenarios", type=int, default=384)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run_experiment(args)
