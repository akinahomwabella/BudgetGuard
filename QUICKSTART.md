# BudgetGuard Quick Start Guide

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python -c "import numpy, pandas, scipy, fastapi; print('All dependencies installed!')"
```

## Running the Examples

### Option 1: Command Line Demo

Run the built-in demonstration:

```bash
python budgetguard.py
```

This will:
- Generate 30/60/90-day forecasts
- Run scenario analysis with spending caps
- Perform variance decomposition
- Compare against baseline models

**Expected output:**
```
======================================================================
BudgetGuard: Cashflow Forecasting & Goal Risk
======================================================================

CASHFLOW FORECASTS
----------------------------------------------------------------------

30-Day Forecast:
  Mean Balance:        $2,234.56
  Median (P50):        $2,189.43
  90% CI:              $1,456.78 to $3,012.34
  Overdraft Risk:      2.34%
  Goal Miss Risk:      45.67%
...
```

### Option 2: REST API

Start the FastAPI server:

```bash
python api.py
```

Or using uvicorn:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Visit the interactive API documentation:
```
http://localhost:8000/docs
```

**Test the API:**

```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_balance": 5000,
    "monthly_income": 4500,
    "categories": [
      {"name": "housing", "mean": 1500, "std": 50},
      {"name": "groceries", "mean": 400, "std": 100}
    ],
    "forecast_days": 60,
    "savings_goal": 3000
  }'
```

### Option 3: Jupyter Notebook

Launch the interactive notebook:

```bash
jupyter notebook demo_notebook.ipynb
```

This provides:
- Step-by-step examples
- Visualizations
- Editable code cells

### Option 4: Web Dashboard

1. Start the API server (see Option 2)
2. Open `dashboard.html` in your browser
3. Adjust parameters and click "Run Forecast"

## Quick Examples

### Example 1: Basic Forecast

```python
from budgetguard import *

# Create model
categories = {
    'housing': ExpenseCategory('housing', mean=1500, std=50, alpha=2, beta=2),
    'groceries': ExpenseCategory('groceries', mean=400, std=100, alpha=2, beta=2)
}
model = BayesianExpenseModel(categories)

# Run simulation
simulator = CashflowSimulator(
    initial_balance=5000,
    expense_model=model,
    monthly_income=4500
)

forecast = simulator.simulate(days=60)
print(f"Expected balance: ${forecast.mean_balance:,.2f}")
print(f"Overdraft risk: {forecast.overdraft_probability:.2%}")
```

### Example 2: Scenario Analysis

```python
analyzer = ScenarioAnalyzer(simulator)

# Test spending caps
results = analyzer.analyze_cap_impact(
    days=60,
    category_caps={
        'dining': [200, 250],
        'entertainment': [150]
    }
)

print(results.sort_values('risk_reduction', ascending=False))
```

### Example 3: Multi-Horizon Forecast

```python
for days in [30, 60, 90]:
    forecast = simulator.simulate(days=days)
    print(f"\n{days}-day forecast:")
    print(f"  Mean: ${forecast.mean_balance:,.2f}")
    print(f"  Risk: {forecast.overdraft_probability:.2%}")
```

## Running Tests

Execute the test suite:

```bash
python test_budgetguard.py
```

Or with verbose output:

```bash
python test_budgetguard.py -v
```

**Expected output:**
```
test_analyze_cap_impact_includes_baseline ... ok
test_analyze_cap_impact_multiple_caps ... ok
test_initialization ... ok
...
----------------------------------------------------------------------
Ran 20 tests in 5.432s

OK
```

## Customizing Parameters

### Expense Categories

Adjust the mean and standard deviation to match your spending:

```python
ExpenseCategory(
    name='dining',
    mean=300.0,      # Average monthly spending
    std=150.0,       # Higher std = more variable
    alpha=2.0,       # Bayesian prior (usually keep at 2.0)
    beta=2.0         # Bayesian prior (usually keep at 2.0)
)
```

**Tips:**
- Fixed expenses (rent, subscriptions): Low std (10-20% of mean)
- Variable expenses (groceries): Medium std (25-50% of mean)
- Discretionary spending (dining, entertainment): High std (50-100% of mean)

### Simulation Parameters

```python
forecast = simulator.simulate(
    days=60,               # Forecast horizon
    n_simulations=10000,   # More = more accurate but slower
    savings_goal=3000,     # Optional target balance
    scenario_caps={'dining': 250}  # Optional spending caps
)
```

## Interpreting Results

### Overdraft Risk
Probability that your balance will go negative:
- **< 5%**: Low risk
- **5-15%**: Moderate risk
- **> 15%**: High risk - consider reducing spending or increasing income

### Goal Miss Risk
Probability of not reaching your savings goal:
- Lower is better
- Use scenario analysis to find spending caps that reduce this risk

### Percentiles
- **P5**: Worst-case scenario (95% chance balance will be above this)
- **P50**: Median outcome (50/50 chance)
- **P95**: Best-case scenario (95% chance balance will be below this)

### Variance Decomposition
Shows which categories contribute most to uncertainty:
- Focus spending reductions on high-variance categories
- These give you the most "bang for your buck" in risk reduction

## Troubleshooting

### Issue: "Module not found"
**Solution:** Install dependencies: `pip install -r requirements.txt`

### Issue: API returns connection error
**Solution:** Make sure the API server is running: `python api.py`

### Issue: Simulations are slow
**Solution:** Reduce `n_simulations` from 10000 to 1000 for faster results

### Issue: Unrealistic forecasts
**Solution:** Check your expense categories - ensure mean and std are realistic for monthly spending

## Next Steps

1. **Customize categories** to match your actual spending
2. **Run backtests** on your historical data to validate accuracy
3. **Experiment with scenarios** to find optimal spending strategies
4. **Integrate with bank APIs** to automate data collection
5. **Build custom visualizations** using the simulation data

## Resources

- **Documentation**: See README.md for full details
- **API Reference**: Visit http://localhost:8000/docs when API is running
- **Examples**: Check demo_notebook.ipynb for more examples
- **Tests**: See test_budgetguard.py for usage patterns

## Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your Python version is 3.7+
3. Review the error messages carefully
4. Consult the README.md for detailed documentation

---

**Happy forecasting!** 🎯
