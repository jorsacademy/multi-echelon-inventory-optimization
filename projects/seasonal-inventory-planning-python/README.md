# Seasonal Inventory Planning in Python

This repository presents a reproducible inventory-planning study for seasonal retail products. It combines synthetic data generation, deterministic inventory optimization, lead-time planning, safety stock, reorder-point calculations, and a single-season newsvendor model.

The project was designed to correct several common modeling mistakes in simplified EOQ implementations. In particular:

- Purchase cost is treated correctly when unit cost is independent of order quantity.
- Integer replenishment quantities are evaluated explicitly instead of applying a gradient-based optimizer to a discontinuous objective.
- Lead time and demand variability are used in reorder-point and safety-stock calculations.
- Seasonal products are also analyzed with a newsvendor model, where overage and underage costs matter directly.

## Project Structure

```text
seasonal-inventory-planning-python/
├── README.md
├── LICENSE.md
├── requirements.txt
├── .gitignore
├── data/
│   └── seasonal_inventory_data.csv
├── src/
│   ├── generate_data.py
│   └── optimize_inventory.py
└── results/
```

## Methods

### 1. Synthetic Data Generation

The dataset contains product-level parameters such as forecast demand, demand standard deviation, ordering cost, holding cost, unit cost, selling price, salvage value, season length, lead time, and target service level.

### 2. Deterministic EOQ Benchmark

For an item with seasonal demand `D`, fixed ordering cost `K`, and seasonal holding cost `h`, the classical EOQ benchmark is:

```text
EOQ = sqrt(2KD / h)
```

Because practical order quantities are integer-valued and the actual number of orders is discrete, the implementation also evaluates feasible integer order quantities directly.

### 3. Reorder Point and Safety Stock

Assuming weekly demand is approximately normal and independent across weeks:

```text
Expected lead-time demand = weekly mean demand × lead time
Safety stock = z × weekly demand standard deviation × sqrt(lead time)
Reorder point = expected lead-time demand + safety stock
```

### 4. Newsvendor Model

For genuinely seasonal products with a single primary ordering decision, the newsvendor critical fractile is:

```text
Critical fractile = Cu / (Cu + Co)
```

where:

- `Cu = selling price - unit cost` is the underage cost.
- `Co = unit cost - salvage value` is the overage cost.

The corresponding normal-demand order quantity is computed from the forecast mean and standard deviation.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset:

```bash
python src/generate_data.py
```

Run the optimization study:

```bash
python src/optimize_inventory.py
```

The optimization script writes result tables to the `results/` directory.

## Important Modeling Notes

The deterministic replenishment model and the newsvendor model answer different business questions. EOQ is appropriate for repeated replenishment under relatively stable demand assumptions, whereas the newsvendor formulation is more appropriate when the business makes one primary purchase decision for a short selling season and leftover inventory loses value.

The generated data are synthetic and intended for educational and research use only. They should not be interpreted as operational recommendations for a real retailer.

## License

This project is provided under a custom non-commercial license. Commercial use, commercial redistribution, paid consulting use, incorporation into commercial products, and other for-profit exploitation are prohibited without separate written permission. See `LICENSE.md` for the full terms.
