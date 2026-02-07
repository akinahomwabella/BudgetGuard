"""
BudgetGuard: Cashflow Forecasting & Goal Risk Assessment System

Implements Bayesian expense modeling and Monte Carlo simulation to:
- Model 30/60/90-day cashflow distributions
- Quantify overdraft and savings-goal risk
- Backtest forecast accuracy and probability calibration
- Perform scenario analysis for spending caps
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExpenseCategory:
    """Represents a spending category with Bayesian parameters."""
    name: str
    mean: float  # Average monthly spending
    std: float   # Standard deviation
    alpha: float  # Beta distribution shape parameter (prior)
    beta: float   # Beta distribution shape parameter (prior)
    observations: int = 0  # Number of historical observations
    
    def update_posterior(self, new_data: List[float]):
        """Update Bayesian parameters with new observations."""
        self.observations += len(new_data)
        if len(new_data) > 0:
            self.mean = np.mean(new_data)
            self.std = np.std(new_data)


@dataclass
class CashflowForecast:
    """Results from cashflow simulation."""
    days: int
    mean_balance: float
    percentile_5: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    overdraft_probability: float
    goal_miss_probability: float
    simulations: np.ndarray


class BayesianExpenseModel:
    """Bayesian model for expense prediction with uncertainty quantification."""
    
    def __init__(self, categories: Dict[str, ExpenseCategory]):
        self.categories = categories
        
    def sample_expenses(self, 
                       days: int, 
                       n_simulations: int = 10000) -> Dict[str, np.ndarray]:
        """
        Generate Monte Carlo samples for expenses over specified days.
        
        Returns dict of category -> array of shape (n_simulations, days)
        """
        samples = {}
        
        for cat_name, category in self.categories.items():
            # Daily expense rate (assuming monthly mean)
            daily_mean = category.mean / 30
            daily_std = category.std / np.sqrt(30)
            
            # Sample daily expenses with gamma distribution (positive values)
            # Convert to gamma parameters
            shape = (daily_mean / daily_std) ** 2
            scale = daily_std ** 2 / daily_mean
            
            daily_samples = np.random.gamma(
                shape=shape,
                scale=scale,
                size=(n_simulations, days)
            )
            
            # Accumulate over days
            samples[cat_name] = np.cumsum(daily_samples, axis=1)
            
        return samples
    
    def apply_scenario(self, 
                      samples: Dict[str, np.ndarray],
                      caps: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Apply spending caps to sampled expenses."""
        capped_samples = {}
        
        for cat_name, cat_samples in samples.items():
            if cat_name in caps:
                cap = caps[cat_name]
                capped_samples[cat_name] = np.minimum(cat_samples, cap)
            else:
                capped_samples[cat_name] = cat_samples
                
        return capped_samples


class CashflowSimulator:
    """Monte Carlo simulator for cashflow forecasting."""
    
    def __init__(self, 
                 initial_balance: float,
                 expense_model: BayesianExpenseModel,
                 monthly_income: float = 0.0):
        self.initial_balance = initial_balance
        self.expense_model = expense_model
        self.monthly_income = monthly_income
        
    def simulate(self, 
                days: int,
                n_simulations: int = 10000,
                savings_goal: Optional[float] = None,
                scenario_caps: Optional[Dict[str, float]] = None) -> CashflowForecast:
        """
        Run Monte Carlo simulation for cashflow over specified days.
        
        Args:
            days: Number of days to forecast
            n_simulations: Number of Monte Carlo samples
            savings_goal: Target savings balance
            scenario_caps: Optional spending caps by category
            
        Returns:
            CashflowForecast with distribution statistics
        """
        # Sample expenses for each category
        expense_samples = self.expense_model.sample_expenses(days, n_simulations)
        
        # Apply scenario caps if provided
        if scenario_caps:
            expense_samples = self.expense_model.apply_scenario(
                expense_samples, scenario_caps
            )
        
        # Sum all expenses across categories
        total_expenses = np.zeros((n_simulations, days))
        for cat_samples in expense_samples.values():
            total_expenses += cat_samples
        
        # Model income (linear over time period)
        income_per_day = self.monthly_income / 30
        cumulative_income = np.arange(1, days + 1) * income_per_day
        
        # Calculate cashflow: initial + income - expenses
        cashflow = (self.initial_balance + 
                   cumulative_income[np.newaxis, :] - 
                   total_expenses)
        
        # Get final day balances
        final_balances = cashflow[:, -1]
        
        # Calculate statistics
        mean_balance = np.mean(final_balances)
        percentiles = np.percentile(final_balances, [5, 25, 50, 75, 95])
        
        # Calculate risk metrics
        overdraft_prob = np.mean(final_balances < 0)
        
        if savings_goal is not None:
            goal_miss_prob = np.mean(final_balances < savings_goal)
        else:
            goal_miss_prob = 0.0
        
        return CashflowForecast(
            days=days,
            mean_balance=mean_balance,
            percentile_5=percentiles[0],
            percentile_25=percentiles[1],
            percentile_50=percentiles[2],
            percentile_75=percentiles[3],
            percentile_95=percentiles[4],
            overdraft_probability=overdraft_prob,
            goal_miss_probability=goal_miss_prob,
            simulations=final_balances
        )


class ForecastBacktester:
    """Backtest forecast accuracy and probability calibration."""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        Args:
            historical_data: DataFrame with columns ['date', 'balance', 'category', 'amount']
        """
        self.historical_data = historical_data
        
    def backtest_accuracy(self, 
                         simulator: CashflowSimulator,
                         test_periods: List[Tuple[datetime, int]]) -> pd.DataFrame:
        """
        Backtest forecast accuracy against historical data.
        
        Args:
            simulator: Configured CashflowSimulator
            test_periods: List of (start_date, forecast_days) tuples
            
        Returns:
            DataFrame with forecast vs actual comparisons
        """
        results = []
        
        for start_date, days in test_periods:
            # Get actual ending balance
            end_date = start_date + timedelta(days=days)
            actual_balance = self._get_actual_balance(end_date)
            
            # Run forecast simulation
            forecast = simulator.simulate(days, n_simulations=5000)
            
            # Calculate errors
            point_error = forecast.mean_balance - actual_balance
            in_confidence_interval = (actual_balance >= forecast.percentile_5 and 
                                     actual_balance <= forecast.percentile_95)
            
            results.append({
                'start_date': start_date,
                'forecast_days': days,
                'forecast_mean': forecast.mean_balance,
                'forecast_p50': forecast.percentile_50,
                'actual_balance': actual_balance,
                'error': point_error,
                'abs_error': abs(point_error),
                'pct_error': (point_error / actual_balance * 100) if actual_balance != 0 else 0,
                'in_90_ci': in_confidence_interval
            })
        
        return pd.DataFrame(results)
    
    def calibration_analysis(self, backtest_results: pd.DataFrame) -> Dict:
        """Analyze probability calibration of forecasts."""
        return {
            'mean_absolute_error': backtest_results['abs_error'].mean(),
            'mean_pct_error': backtest_results['pct_error'].mean(),
            'rmse': np.sqrt((backtest_results['error'] ** 2).mean()),
            'confidence_coverage': backtest_results['in_90_ci'].mean(),
            'expected_coverage': 0.90
        }
    
    def _get_actual_balance(self, date: datetime) -> float:
        """Get actual balance on specific date from historical data."""
        # This is a simplified version - in practice would query actual data
        mask = self.historical_data['date'] <= date
        return self.historical_data[mask]['balance'].iloc[-1] if mask.any() else 0.0


class BaselineComparator:
    """Compare against deterministic baseline models."""
    
    @staticmethod
    def moving_average_forecast(historical_balances: List[float], 
                                window: int = 30) -> float:
        """Simple moving average baseline."""
        if len(historical_balances) < window:
            window = len(historical_balances)
        return np.mean(historical_balances[-window:])
    
    @staticmethod
    def rules_based_budget(monthly_income: float, 
                          savings_rate: float = 0.2) -> float:
        """Simple rules-based budget (50/30/20 rule)."""
        return monthly_income * savings_rate


class ScenarioAnalyzer:
    """Analyze risk reduction from spending caps and identify variance drivers."""
    
    def __init__(self, simulator: CashflowSimulator):
        self.simulator = simulator
        
    def analyze_cap_impact(self, 
                          days: int,
                          category_caps: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Analyze impact of different spending caps on risk metrics.
        
        Args:
            days: Forecast horizon
            category_caps: Dict of category -> list of cap values to test
            
        Returns:
            DataFrame with risk metrics for each scenario
        """
        results = []
        
        # Baseline (no caps)
        baseline = self.simulator.simulate(days, n_simulations=5000)
        results.append({
            'scenario': 'baseline',
            'category': 'none',
            'cap': None,
            'overdraft_prob': baseline.overdraft_probability,
            'mean_balance': baseline.mean_balance,
            'p5_balance': baseline.percentile_5
        })
        
        # Test each category cap
        for category, cap_values in category_caps.items():
            for cap in cap_values:
                scenario_caps = {category: cap}
                forecast = self.simulator.simulate(
                    days, 
                    n_simulations=5000,
                    scenario_caps=scenario_caps
                )
                
                results.append({
                    'scenario': f'{category}_cap_{cap}',
                    'category': category,
                    'cap': cap,
                    'overdraft_prob': forecast.overdraft_probability,
                    'mean_balance': forecast.mean_balance,
                    'p5_balance': forecast.percentile_5,
                    'risk_reduction': (baseline.overdraft_probability - 
                                     forecast.overdraft_probability)
                })
        
        return pd.DataFrame(results)
    
    def variance_decomposition(self, days: int) -> pd.DataFrame:
        """
        Identify primary contributors to cashflow variance.
        
        Uses ANOVA-style variance decomposition.
        """
        # Run baseline simulation
        n_sims = 5000
        expense_samples = self.simulator.expense_model.sample_expenses(
            days, n_sims
        )
        
        # Calculate variance contribution of each category
        results = []
        total_expenses = np.zeros((n_sims, days))
        
        for cat_name, cat_samples in expense_samples.items():
            total_expenses += cat_samples
            
            # Variance at final day
            final_variance = np.var(cat_samples[:, -1])
            
            results.append({
                'category': cat_name,
                'variance': final_variance,
                'std_dev': np.sqrt(final_variance),
                'mean': np.mean(cat_samples[:, -1])
            })
        
        df = pd.DataFrame(results)
        total_var = df['variance'].sum()
        df['variance_contribution_pct'] = (df['variance'] / total_var * 100)
        
        return df.sort_values('variance_contribution_pct', ascending=False)


def create_example_model() -> BayesianExpenseModel:
    """Create example expense model with typical spending categories."""
    categories = {
        'housing': ExpenseCategory(
            name='housing',
            mean=1500.0,
            std=50.0,
            alpha=2.0,
            beta=2.0
        ),
        'groceries': ExpenseCategory(
            name='groceries',
            mean=400.0,
            std=100.0,
            alpha=2.0,
            beta=2.0
        ),
        'dining': ExpenseCategory(
            name='dining',
            mean=300.0,
            std=150.0,
            alpha=2.0,
            beta=2.0
        ),
        'transportation': ExpenseCategory(
            name='transportation',
            mean=200.0,
            std=80.0,
            alpha=2.0,
            beta=2.0
        ),
        'subscriptions': ExpenseCategory(
            name='subscriptions',
            mean=100.0,
            std=20.0,
            alpha=2.0,
            beta=2.0
        ),
        'entertainment': ExpenseCategory(
            name='entertainment',
            mean=200.0,
            std=120.0,
            alpha=2.0,
            beta=2.0
        )
    }
    
    return BayesianExpenseModel(categories)


def main():
    """Example usage and demonstration."""
    print("=" * 70)
    print("BudgetGuard: Cashflow Forecasting & Goal Risk")
    print("=" * 70)
    print()
    
    # Create expense model
    expense_model = create_example_model()
    
    # Initialize simulator
    simulator = CashflowSimulator(
        initial_balance=5000.0,
        expense_model=expense_model,
        monthly_income=4500.0
    )
    
    # Run forecasts for 30/60/90 days
    print("CASHFLOW FORECASTS")
    print("-" * 70)
    
    for days in [30, 60, 90]:
        forecast = simulator.simulate(
            days=days,
            n_simulations=10000,
            savings_goal=3000.0
        )
        
        print(f"\n{days}-Day Forecast:")
        print(f"  Mean Balance:        ${forecast.mean_balance:,.2f}")
        print(f"  Median (P50):        ${forecast.percentile_50:,.2f}")
        print(f"  90% CI:              ${forecast.percentile_5:,.2f} to ${forecast.percentile_95:,.2f}")
        print(f"  Overdraft Risk:      {forecast.overdraft_probability:.2%}")
        print(f"  Goal Miss Risk:      {forecast.goal_miss_probability:.2%}")
    
    # Scenario Analysis
    print("\n\nSCENARIO ANALYSIS: Impact of Spending Caps")
    print("-" * 70)
    
    analyzer = ScenarioAnalyzer(simulator)
    
    scenario_results = analyzer.analyze_cap_impact(
        days=60,
        category_caps={
            'dining': [200, 250],
            'entertainment': [100, 150],
            'subscriptions': [75, 90]
        }
    )
    
    print("\nRisk Reduction from Category Caps:")
    for _, row in scenario_results[scenario_results['scenario'] != 'baseline'].iterrows():
        if 'risk_reduction' in row:
            print(f"  {row['category']:<15} cap ${row['cap']:>6,.0f}: "
                  f"Overdraft risk {row['overdraft_prob']:.2%} "
                  f"(reduction: {row['risk_reduction']:.2%})")
    
    # Variance Decomposition
    print("\n\nVARIANCE DECOMPOSITION: Primary Contributors")
    print("-" * 70)
    
    variance_results = analyzer.variance_decomposition(days=60)
    print("\nContribution to Cashflow Uncertainty:")
    for _, row in variance_results.iterrows():
        print(f"  {row['category']:<15}: {row['variance_contribution_pct']:>5.1f}% "
              f"(σ = ${row['std_dev']:,.0f})")
    
    # Baseline Comparison
    print("\n\nBASELINE COMPARISONS")
    print("-" * 70)
    
    historical_balances = [4800, 4900, 5100, 4700, 5000]
    ma_forecast = BaselineComparator.moving_average_forecast(historical_balances)
    rules_budget = BaselineComparator.rules_based_budget(4500)
    
    print(f"\n  Moving Average Baseline:     ${ma_forecast:,.2f}")
    print(f"  Rules-Based Budget (20%):    ${rules_budget:,.2f}")
    print(f"  Bayesian Model (60-day P50): ${simulator.simulate(60).percentile_50:,.2f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
