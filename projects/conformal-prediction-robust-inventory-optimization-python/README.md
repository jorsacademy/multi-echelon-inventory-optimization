# Conformal Prediction + Robust Inventory Optimization

A decision-making-under-uncertainty project that combines simultaneous split conformal prediction with an exact small-horizon box-robust inventory LP.

The Industrial Engineering / Operations Research pipeline is:

```text
contextual demand data
-> ExtraTrees or optional Bayesian multi-period demand predictor
-> proper training / calibration split
-> simultaneous conformal demand trajectory box
-> exact robust multi-period procurement LP
-> held-out inventory cost / fill rate / stockout evaluation
```

The repository separates three questions:

1. How accurate is the demand predictor?
2. Does the uncertainty band attain the intended statistical coverage?
3. What operational decisions result from optimizing against that band?

A conformal coverage statement is not relabeled as a service-level guarantee.

## Planning problem

The planner commits to orders `q_t` for a short multi-period horizon before the demand trajectory is observed.

```text
inventory_t =
    initial_inventory
    + cumulative_orders_t
    - cumulative_demand_t
```

Positive inventory incurs holding cost; negative inventory represents backlog/shortage. The plan trades procurement cost, holding cost, and shortage cost subject to period-specific order capacities.

The benchmark is a pre-commitment problem. It does not contain adaptive recourse after demand is revealed.

## Synthetic contextual demand model

Each sample contains a complete demand trajectory plus planning-time context. The hidden generator includes nonlinear context effects, seasonality, heteroskedastic noise, serial correlation, common shocks, and occasional positive demand surges.

The predictor never receives the hidden generator. The synthetic setup is used so out-of-sample experiments and an information-advantaged distribution reference can be built without claiming access to real industrial demand data.

## Demand predictors

The default point predictor remains a multi-output `ExtraTreesRegressor`:

```text
x -> [d_hat_1, ..., d_hat_H]
```

An optional PyTorch Bayesian neural network is also available:

```text
context
  -> mean-field Bayesian MLP
  -> posterior weight samples
  -> posterior-mean demand forecast
  -> split conformal calibration
  -> robust inventory optimization
```

The BNN uses reparameterized Gaussian weights, an analytic Gaussian KL term, and a Monte Carlo ELBO that evaluates the likelihood separately for each posterior weight draw before averaging. Inputs and multi-period demand targets are standardized on the proper-training split. Its posterior mean is deliberately passed through the same conformal and robust-optimization layers as ExtraTrees so downstream comparisons remain controlled.

The BNN also exposes epistemic standard deviation from posterior weight sampling. That quantity is reported separately from conformal coverage: a Bayesian posterior interval and a split-conformal coverage statement are not treated as interchangeable guarantees.

ExtraTrees is not claimed to be the best forecasting architecture, and the BNN is not claimed to dominate it. The repository is structured so predictive accuracy, uncertainty calibration, and downstream decision quality can be compared separately.

## Simultaneous split conformal prediction

The data are split into:

```text
proper training
calibration
held-out test
```

The predictor is fitted only on the proper-training split. A period scale `s_t > 0` is also estimated only from proper-training residuals.

For calibration trajectory `i`:

```text
score_i = max_t |d_i,t - d_hat_i,t| / s_t
```

For calibration size `n`, the conformal radius uses the order statistic at:

```text
ceil((n + 1) * (1 - alpha))
```

The resulting box is:

```text
max(d_hat_t - q*s_t, 0) <= d_t <= d_hat_t + q*s_t
```

Because a single maximum score is calibrated for the whole trajectory, the target is simultaneous trajectory coverage rather than only periodwise marginal coverage.

### What the guarantee means

Under the standard exchangeability assumptions, split conformal provides a finite-sample marginal coverage statement for a new trajectory.

It does not imply:

- conditional coverage for every context;
- robustness to arbitrary distribution shift;
- that every finite held-out sample must empirically exceed the nominal level;
- an inventory fill-rate guarantee;
- a stockout-probability guarantee.

Coverage and operational service metrics are reported separately throughout the project.

## Comparison uncertainty sets

The experiment compares:

- `Deterministic`: optimize against the point forecast only.
- `Classical sigma robust`: proper-training residual standard deviations with a fixed multiplier.
- `Marginal quantile robust`: periodwise absolute-residual quantiles.
- `Conformal robust`: one trajectory-level max-normalized conformal score.

The marginal quantile construction may have strong pointwise coverage while showing much lower joint trajectory coverage because every horizon period must be covered simultaneously.

## Exact box-robust inventory LP

For an uncertainty box:

```text
lower_t <= demand_t <= upper_t
```

the implementation enumerates all `2^H` vertices.

For each vertex scenario `s` and period `t`:

```text
h[s,t] - b[s,t]
=
initial_inventory
+ cumulative_orders_t
- cumulative_demand[s,t]
```

with `h[s,t] >= 0` and `b[s,t] >= 0`.

An epigraph variable `z` bounds holding plus shortage cost for every vertex. The robust objective is procurement cost plus `z`.

For fixed orders, the piecewise-linear holding/backlog penalty is convex in demand. Its maximum over a rectangular uncertainty set is attained at a vertex. Therefore vertex enumeration gives the exact worst case for the declared continuous box-robust model.

This exactness statement is intentionally small-horizon: vertex enumeration scales as `2^H` and is not presented as a long-horizon robust-optimization architecture.

The LPs are solved with SciPy/HiGHS.

## Independent optimization checks

The regression suite does not trust the LP formulation alone.

A one-period robust fixture is checked by:

1. solving the robust LP;
2. enumerating 20,001 feasible order quantities on an independent dense grid;
3. evaluating the worst endpoint cost directly;
4. requiring the LP optimum to match the grid optimum to tolerance.

A second test evaluates every uncertainty-box vertex at the returned robust plan and verifies the epigraph worst-case cost.

The SAA formulation is also checked against a fixed alternative on the same finite scenario set.

## Operational metrics

For each realized demand trajectory the evaluator reports:

- total procurement plus inventory/backlog cost;
- immediate fill rate;
- stockout-period rate;
- total ordered quantity;
- relative cost gap versus an information-advantaged realized-demand plan.

Backlogged demand consumes later inventory before new-period demand is counted as immediately filled.

## Information-advantaged references

### Clairvoyant realized-demand plan

For each held-out realized trajectory, a deterministic LP is solved using the actual future demand. This is used only as a reference and is not a deployable policy.

### True-distribution SAA reference

Because the demand model is synthetic, a small reference experiment can sample the hidden conditional generator:

```text
context
-> hidden conditional demand generator
-> finite scenario sample
-> expected-cost SAA inventory LP
-> independent Monte Carlo evaluation
```

Practical policies do not receive this generator. The SAA control is information-advantaged and finite-sample; it is not called an exact stochastic optimum or a lower bound.

## Development benchmark

Seed-42 development configuration:

```text
horizon                         6
proper-training trajectories  700
calibration trajectories      250
test trajectories             700
inventory planning contexts    80
ExtraTrees                    200 trees
alpha                           0.10
```

Prediction RMSE:

```text
train          6.796
calibration   11.301
test          10.601
```

Coverage results:

```text
                               trajectory   pointwise   mean width
simultaneous conformal            89.1%       97.5%       49.596
marginal residual quantile        73.3%       92.4%       37.976
classical 2-sigma                 44.6%         --        27.122
```

The conformal normalized radius was `3.660`. The observed 89.1% coverage on one finite test sample is not interpreted as a contradiction of a 90% split-conformal marginal coverage statement.

Held-out inventory decisions:

```text
method                       mean cost   fill rate   stockout   order qty
Deterministic                  2535.30      0.8846     0.4042      374.26
Classical sigma robust         2430.50      0.9866     0.0750      414.68
Marginal quantile robust       2556.37      0.9953     0.0271      431.81
Conformal robust               2727.92      0.9974     0.0187      451.33
```

The conformal robust policy bought stronger service protection but at higher cost. That is the intended robustness-versus-conservatism trade-off, not an implementation failure.

## Validated GitHub Actions run

GitHub Actions run `33117888742` completed successfully on Ubuntu 24.04 / CPython 3.12.14 with:

```text
NumPy          2.5.2
SciPy          1.18.1
scikit-learn   1.9.0
```

All 11 regression/oracle tests passed.

CI smoke demand-band results:

```text
target trajectory coverage          90.0%
conformal trajectory coverage        97.5%
marginal-box trajectory coverage     85.0%
2-sigma trajectory coverage          48.3%
conformal pointwise coverage         99.5%
marginal-box pointwise coverage      95.2%
mean conformal band width             64.398
conformal normalized radius            4.217
```

CI smoke inventory results:

```text
method                       mean cost   fill rate   stockout   order qty
Deterministic                  2252.73      0.8819     0.5167      336.45
Classical sigma robust         2127.41      0.9944     0.0167      379.43
Marginal quantile robust       2312.08      0.9989     0.0167      404.84
Conformal robust               2637.93      1.0000     0.0000      448.70
```

Two-context information-advantaged distribution reference:

```text
method                       MC mean cost   fill rate   stockout   order qty
Distribution-aware SAA           2045.73      0.9611     0.1461      348.86
Deterministic                    2939.82      0.6983     0.7828      314.66
Classical sigma robust           2184.02      0.9423     0.2875      357.59
Marginal quantile robust         2181.95      0.9807     0.1195      380.73
Conformal robust                 2443.48      0.9974     0.0328      426.88
```

The conformal robust plan provided the strongest service protection in this short CI smoke, but at higher expected cost. No universal dominance claim is made.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

With the optional PyTorch dependency installed, the executable suite reports 14 tests. In addition to the original deterministic-generation, conformal, optimization, and end-to-end checks, it verifies the analytic Gaussian KL, Monte Carlo ELBO construction, multi-period BNN fitting, nonnegative predictions, epistemic standard deviations, and reproducible posterior-mean predictions.

## Run the experiment

Default ExtraTrees path:

```bash
pip install -r requirements.txt
python run_conformal_inventory.py --self-test
python run_conformal_inventory.py \
  --seed 42 \
  --horizon 6 \
  --train-samples 700 \
  --calibration-samples 250 \
  --test-samples 700 \
  --planning-samples 80 \
  --trees 200 \
  --alpha 0.10 \
  --distribution-reference-contexts 5 \
  --saa-scenarios 192 \
  --reference-eval-scenarios 384
```

Optional BNN path:

```bash
pip install -r requirements-bnn.txt
python run_conformal_inventory.py \
  --predictor bnn \
  --seed 42 \
  --horizon 6 \
  --train-samples 700 \
  --calibration-samples 250 \
  --test-samples 700 \
  --planning-samples 80 \
  --bnn-hidden 32 \
  --bnn-steps 600 \
  --bnn-train-samples 3 \
  --bnn-prediction-samples 300 \
  --alpha 0.10
```

The BNN likelihood noise is expressed on the standardized target scale and is configurable with `--bnn-noise-std`. The BNN is an experimental predictor extension; conformal calibration remains the layer that supplies the finite-sample coverage target used by the robust planner.

## Exactness and scope

Exact claims are deliberately narrow:

- HiGHS solves the declared deterministic, SAA, and robust LPs to its reported optimum;
- the box-robust formulation enumerates every vertex of the declared uncertainty box;
- for the declared convex inventory penalty, vertex enumeration gives the exact worst case over that box;
- the one-period robust fixture is independently checked by dense enumeration;
- the SAA objective is the exact sample-average LP objective for the supplied finite scenario set.

Statistical claim:

- the simultaneous conformal construction targets finite-sample marginal trajectory coverage under standard split-conformal exchangeability assumptions.

Not claimed:

- conditional coverage for every feature vector;
- robustness under arbitrary distribution shift;
- equality between conformal coverage and service level;
- that conformal robust planning minimizes expected cost;
- that the synthetic benchmark represents a real company demand process;
- that vertex enumeration scales to long horizons;
- that the finite-sample SAA reference is the exact true-distribution optimum.
