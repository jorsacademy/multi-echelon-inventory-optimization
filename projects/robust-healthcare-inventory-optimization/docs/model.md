# Mathematical model and implementation status

The project models multi-period inventory planning for perishable healthcare products across hospital-type/region segments under uncertain demand.

Core parameters retain the original case values: unit cost 45, monthly holding rate 0.0125, shortage penalty 200, obsolescence penalty 30, fixed order cost 500, three-period regular lead time, 18-period shelf life, 95% target service level, 2.5M initial-inventory budget, and 50,000-unit capacity.

## Implemented baseline

The production code currently implements:

- Monte Carlo demand scenarios with correlated adoption/economic factors and seasonal demand.
- Age-bucket inventory with FEFO consumption and shelf-life obsolescence.
- Explicit regular-order pipeline with configurable lead time.
- Reorder-point/order-quantity heuristic using empirical monthly mean and standard deviation.
- Initial budget/capacity enforcement.
- Expected cost ledger, segment fill-rate service levels, and CVaR(95%) reporting.

## Deliberately not mislabeled as implemented

The original concept also specifies a Wasserstein distributionally robust optimization (DRO) ambiguity set, Benders decomposition, a genetic algorithm for integer decisions, and a chance-style service reliability constraint. Those methods are not mathematically equivalent to the current heuristic baseline and are therefore documented as future extensions rather than represented as completed functionality.

A future exact solver should introduce a tractable Wasserstein dual/reformulation (or a decomposition algorithm with justified ambiguity-set assumptions), explicitly model first-stage versus recourse decisions, and verify the service reliability formulation against out-of-sample scenarios.
