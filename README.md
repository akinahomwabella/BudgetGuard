# BudgetGuard: Cashflow Forecasting & Goal Risk

BudgetGuard is a probabilistic cashflow forecasting system that uses Bayesian expense modeling and Monte Carlo simulation to estimate short-term financial risk. The system simulates future cashflow scenarios to quantify the probability of overdrafts and savings-goal failures across multiple time horizons, and exposes results through a REST API and interactive dashboard.

---

## Key Features

- **Bayesian Expense Modeling**  
  Models spending using category-specific probability distributions to capture uncertainty and variability in expenses.

- **Monte Carlo Cashflow Forecasting**  
  Runs 10,000+ simulations to generate probabilistic forecasts for 30/60/90-day horizons with full confidence interval reporting.

- **Risk Quantification**  
  Computes probabilities of overdrafts and savings-goal misses directly from simulated cashflow distributions.

- **Scenario Analysis & Variance Decomposition**  
  Evaluates the impact of spending caps on financial risk and identifies which expense categories contribute most to overall uncertainty.

- **API-First Design**  
  Forecasting logic is exposed via a FastAPI backend, enabling reuse across dashboards and external clients.

- **Interactive Dashboard & Notebooks**  
  Includes a lightweight HTML dashboard and Jupyter notebooks for exploration, validation, and demonstration.

---

---

## How It Works

1. **Expense Modeling**  
   Historical expenses are modeled using Bayesian, category-specific distributions.

2. **Simulation**  
   Thousands of future cashflow paths are generated via Monte Carlo simulation.

3. **Risk Estimation**  
   Simulated outcomes are used to compute overdraft probabilities, goal-miss risk, and uncertainty bounds.

4. **Scenario Analysis**  
   Spending constraints are applied to evaluate potential reductions in financial risk.

5. **Delivery**  
   Results are served through a REST API and visualized in a simple dashboard.

---

## Running the Project Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Start API
uvicorn api:app --reload

### 3. View the dashboard
python -m http.server 5500




