# Chance-Constrained Inventory Optimization in Python

This repository presents a small but fully specified chance-constrained optimization model for a single-period inventory decision under uncertain demand.

The example is intended for operations research education. No real or fictional company name is used.

## Scenario

A distributor must choose an order quantity `Q` before demand is observed. Demand is uncertain and modeled as normally distributed:

`D ~ Normal(mu, sigma^2)`.

Ordering too much creates leftover inventory. Ordering too little creates unmet demand. In addition to minimizing expected inventory cost, the decision maker requires a minimum service reliability: demand must be fully covered with probability at least `alpha`.

Baseline parameters:

- mean demand: `mu = 100`
- standard deviation: `sigma = 20`
- holding cost per leftover unit: `h = 2`
- shortage cost per unmet unit: `p = 5`
- required service probability: `alpha = 0.95`

## Mathematical model

Decision variable:

`Q >= 0`, the order quantity.

Objective:

`min h E[(Q-D)^+] + p E[(D-Q)^+]`

where `(x)^+ = max(x, 0)`.

Chance constraint:

`P(D <= Q) >= alpha`.

For normally distributed demand, the chance constraint has the exact deterministic equivalent

`Q >= mu + sigma * Phi^{-1}(alpha)`,

where `Phi` is the standard normal cumulative distribution function.

For the baseline values,

`Q >= 100 + 20 * Phi^{-1}(0.95) = 132.8971`.

## Correct expected-cost expressions

Let

`z = (Q - mu) / sigma`.

Then

`E[(Q-D)^+] = sigma * phi(z) + (Q-mu) * Phi(z)`

and

`E[(D-Q)^+] = sigma * phi(z) + (mu-Q) * (1-Phi(z))`,

where `phi` is the standard normal density.

These expressions are required. Replacing random demand by its mean inside `max()` is not an expected-cost calculation.

## Why the chance constraint binds here

Without the chance constraint, the classical single-period critical fractile is

`p / (p+h) = 5/7 = 0.7142857`.

The unconstrained optimizer therefore corresponds to approximately the 71.43rd percentile of demand. The required service probability is 95%, so the chance constraint is stricter and becomes binding.

Hence the constrained optimum is approximately

`Q* = 132.8971`.

At this value,

`P(D <= Q*) = 0.95`.

## Repository contents

- `chance_constrained_inventory.py`: model, analytical benchmark, numerical solution, and verification
- `monte_carlo_validation.py`: out-of-sample simulation of the service probability and expected cost
- `sensitivity_analysis.py`: effect of changing the required reliability level
- `tests/test_model.py`: regression tests for the mathematics and implementation
- `requirements.txt`: Python dependencies
- `LICENSE`: non-commercial educational-use license

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the model

```bash
python chance_constrained_inventory.py
```

Expected optimal order quantity:

`132.8971` approximately.

## Monte Carlo validation

```bash
python monte_carlo_validation.py
```

The empirical service probability should be close to `0.95`. Small deviations are expected because simulation uses a finite random sample.

## Sensitivity analysis

```bash
python sensitivity_analysis.py
```

As `alpha` increases, the deterministic lower bound and the optimal order quantity increase whenever the chance constraint is binding.

## Tests

```bash
pytest -q
```

## Modeling assumptions

The model assumes a single selling period, one item, continuous order quantity, normally distributed demand with known parameters, linear overage and shortage costs, and no fixed ordering cost or capacity limit. The normal distribution permits negative demand in principle; with `mu = 100` and `sigma = 20`, that probability is negligible for this teaching example. For applications where nonnegative demand is essential, a truncated normal, lognormal, empirical, or scenario-based distribution may be more appropriate.

## License

This repository is provided for educational and non-commercial use only. Commercial use is prohibited. See `LICENSE` for the complete terms.
