# Contextual Optimization for the Newsvendor Problem

A research sandbox for **contextual stochastic optimization**: learn decisions that use observable covariates before uncertain demand is realized.

The benchmark compares three contextual decision paradigms:

- **predict-then-optimize:** linear conditional quantile regression followed by the newsvendor critical-fractile decision;
- **direct decision learning:** a neural decision rule trained directly on realized underage/overage cost;
- **local empirical optimization:** k-nearest-neighbor sample-average approximation (kNN-SAA).

All policies receive the same contextual features and are evaluated on out-of-sample downstream decision cost over repeated random seeds.

## Run

```bash
pip install -e ".[dev]"
python scripts/benchmark.py
pytest
```

## Research scope

The repository follows the contextual-optimization umbrella surveyed by Sadana et al., *European Journal of Operational Research* 320(2), 2025. It deliberately separates prediction quality from prescription quality and evaluates policies by downstream operational cost.

This is a transparent synthetic benchmark, not a production inventory system.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
