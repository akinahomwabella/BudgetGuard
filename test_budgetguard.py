"""
Unit tests for BudgetGuard components
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from budgetguard import (
    ExpenseCategory,
    BayesianExpenseModel,
    CashflowSimulator,
    ScenarioAnalyzer,
    ForecastBacktester,
    BaselineComparator
)


class TestExpenseCategory(unittest.TestCase):
    """Test ExpenseCategory functionality."""
    
    def test_initialization(self):
        """Test category creation."""
        cat = ExpenseCategory(
            name="groceries",
            mean=400.0,
            std=100.0,
            alpha=2.0,
            beta=2.0
        )
        self.assertEqual(cat.name, "groceries")
        self.assertEqual(cat.mean, 400.0)
        self.assertEqual(cat.observations, 0)
    
    def test_update_posterior(self):
        """Test Bayesian updating."""
        cat = ExpenseCategory(
            name="test",
            mean=100.0,
            std=20.0,
            alpha=2.0,
            beta=2.0
        )
        
        new_data = [110.0, 105.0, 95.0]
        cat.update_posterior(new_data)
        
        self.assertEqual(cat.observations, 3)
        self.assertAlmostEqual(cat.mean, 103.33, places=1)


class TestBayesianExpenseModel(unittest.TestCase):
    """Test BayesianExpenseModel functionality."""
    
    def setUp(self):
        """Set up test categories."""
        self.categories = {
            'test1': ExpenseCategory(
                name='test1',
                mean=100.0,
                std=10.0,
                alpha=2.0,
                beta=2.0
            ),
            'test2': ExpenseCategory(
                name='test2',
                mean=200.0,
                std=30.0,
                alpha=2.0,
                beta=2.0
            )
        }
        self.model = BayesianExpenseModel(self.categories)
    
    def test_sample_expenses_shape(self):
        """Test that samples have correct shape."""
        samples = self.model.sample_expenses(days=30, n_simulations=1000)
        
        self.assertEqual(len(samples), 2)  # Two categories
        self.assertEqual(samples['test1'].shape, (1000, 30))
        self.assertEqual(samples['test2'].shape, (1000, 30))
    
    def test_sample_expenses_positive(self):
        """Test that all samples are positive."""
        samples = self.model.sample_expenses(days=30, n_simulations=1000)
        
        for cat_name, cat_samples in samples.items():
            self.assertTrue(np.all(cat_samples >= 0))
    
    def test_sample_expenses_cumulative(self):
        """Test that samples are cumulative."""
        samples = self.model.sample_expenses(days=30, n_simulations=100)
        
        for cat_name, cat_samples in samples.items():
            # Check that values generally increase over time
            for sim in cat_samples:
                self.assertTrue(sim[-1] >= sim[0])
    
    def test_apply_scenario(self):
        """Test spending cap application."""
        samples = self.model.sample_expenses(days=30, n_simulations=100)
        caps = {'test1': 50.0}  # Cap first category at $50
        
        capped_samples = self.model.apply_scenario(samples, caps)
        
        # Check that cap is enforced
        self.assertTrue(np.all(capped_samples['test1'] <= 50.0))
        # Check that uncapped category is unchanged
        np.testing.assert_array_equal(
            capped_samples['test2'],
            samples['test2']
        )


class TestCashflowSimulator(unittest.TestCase):
    """Test CashflowSimulator functionality."""
    
    def setUp(self):
        """Set up test simulator."""
        categories = {
            'test': ExpenseCategory(
                name='test',
                mean=300.0,
                std=50.0,
                alpha=2.0,
                beta=2.0
            )
        }
        self.model = BayesianExpenseModel(categories)
        self.simulator = CashflowSimulator(
            initial_balance=1000.0,
            expense_model=self.model,
            monthly_income=500.0
        )
    
    def test_simulate_returns_forecast(self):
        """Test that simulate returns a CashflowForecast."""
        forecast = self.simulator.simulate(days=30, n_simulations=100)
        
        self.assertIsNotNone(forecast.mean_balance)
        self.assertIsNotNone(forecast.overdraft_probability)
        self.assertEqual(forecast.days, 30)
    
    def test_simulate_percentiles_ordered(self):
        """Test that percentiles are in correct order."""
        forecast = self.simulator.simulate(days=30, n_simulations=1000)
        
        self.assertLessEqual(forecast.percentile_5, forecast.percentile_25)
        self.assertLessEqual(forecast.percentile_25, forecast.percentile_50)
        self.assertLessEqual(forecast.percentile_50, forecast.percentile_75)
        self.assertLessEqual(forecast.percentile_75, forecast.percentile_95)
    
    def test_simulate_with_savings_goal(self):
        """Test goal miss probability calculation."""
        forecast = self.simulator.simulate(
            days=30,
            n_simulations=1000,
            savings_goal=500.0
        )
        
        self.assertGreaterEqual(forecast.goal_miss_probability, 0.0)
        self.assertLessEqual(forecast.goal_miss_probability, 1.0)
    
    def test_simulate_with_caps(self):
        """Test simulation with spending caps."""
        forecast_no_cap = self.simulator.simulate(
            days=30,
            n_simulations=100
        )
        
        forecast_with_cap = self.simulator.simulate(
            days=30,
            n_simulations=100,
            scenario_caps={'test': 100.0}  # Very low cap
        )
        
        # With cap, final balance should generally be higher
        self.assertGreater(
            forecast_with_cap.mean_balance,
            forecast_no_cap.mean_balance
        )
    
    def test_overdraft_probability_range(self):
        """Test that overdraft probability is valid."""
        forecast = self.simulator.simulate(days=30, n_simulations=1000)
        
        self.assertGreaterEqual(forecast.overdraft_probability, 0.0)
        self.assertLessEqual(forecast.overdraft_probability, 1.0)


class TestScenarioAnalyzer(unittest.TestCase):
    """Test ScenarioAnalyzer functionality."""
    
    def setUp(self):
        """Set up test analyzer."""
        categories = {
            'cat1': ExpenseCategory(
                name='cat1',
                mean=300.0,
                std=100.0,
                alpha=2.0,
                beta=2.0
            ),
            'cat2': ExpenseCategory(
                name='cat2',
                mean=200.0,
                std=80.0,
                alpha=2.0,
                beta=2.0
            )
        }
        model = BayesianExpenseModel(categories)
        simulator = CashflowSimulator(
            initial_balance=1000.0,
            expense_model=model,
            monthly_income=500.0
        )
        self.analyzer = ScenarioAnalyzer(simulator)
    
    def test_analyze_cap_impact_includes_baseline(self):
        """Test that analysis includes baseline scenario."""
        results = self.analyzer.analyze_cap_impact(
            days=30,
            category_caps={'cat1': [200.0]}
        )
        
        baseline = results[results['scenario'] == 'baseline']
        self.assertEqual(len(baseline), 1)
    
    def test_analyze_cap_impact_multiple_caps(self):
        """Test analysis with multiple cap values."""
        results = self.analyzer.analyze_cap_impact(
            days=30,
            category_caps={
                'cat1': [200.0, 250.0],
                'cat2': [150.0]
            }
        )
        
        # Should have baseline + 3 scenarios
        self.assertEqual(len(results), 4)
    
    def test_variance_decomposition(self):
        """Test variance decomposition."""
        variance_df = self.analyzer.variance_decomposition(days=30)
        
        # Should have entry for each category
        self.assertEqual(len(variance_df), 2)
        
        # Contributions should sum to ~100%
        total_contribution = variance_df['variance_contribution_pct'].sum()
        self.assertAlmostEqual(total_contribution, 100.0, places=0)
        
        # All values should be positive
        self.assertTrue(np.all(variance_df['variance'] > 0))
        self.assertTrue(np.all(variance_df['std_dev'] > 0))


class TestBaselineComparator(unittest.TestCase):
    """Test BaselineComparator functionality."""
    
    def test_moving_average_forecast(self):
        """Test moving average baseline."""
        balances = [1000, 1020, 980, 1010, 990]
        forecast = BaselineComparator.moving_average_forecast(balances, window=3)
        
        # Should be average of last 3 values
        expected = np.mean([980, 1010, 990])
        self.assertAlmostEqual(forecast, expected)
    
    def test_moving_average_short_window(self):
        """Test moving average with insufficient data."""
        balances = [1000, 1020]
        forecast = BaselineComparator.moving_average_forecast(balances, window=5)
        
        # Should use all available data
        expected = np.mean([1000, 1020])
        self.assertAlmostEqual(forecast, expected)
    
    def test_rules_based_budget(self):
        """Test rules-based budget calculation."""
        monthly_income = 4500.0
        savings = BaselineComparator.rules_based_budget(
            monthly_income,
            savings_rate=0.20
        )
        
        self.assertEqual(savings, 900.0)


class TestForecastBacktester(unittest.TestCase):
    """Test ForecastBacktester functionality."""
    
    def setUp(self):
        """Set up test backtester."""
        # Create synthetic historical data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        self.historical_data = pd.DataFrame({
            'date': dates,
            'balance': 1000.0 + np.cumsum(np.random.normal(0, 10, len(dates))),
            'category': 'test',
            'amount': np.random.normal(50, 10, len(dates))
        })
        self.backtester = ForecastBacktester(self.historical_data)
    
    def test_calibration_analysis_keys(self):
        """Test that calibration analysis returns expected keys."""
        mock_results = pd.DataFrame({
            'abs_error': [10, 20, 15],
            'pct_error': [5, 10, 7.5],
            'error': [10, -20, 15],
            'in_90_ci': [True, False, True]
        })
        
        calibration = self.backtester.calibration_analysis(mock_results)
        
        self.assertIn('mean_absolute_error', calibration)
        self.assertIn('mean_pct_error', calibration)
        self.assertIn('rmse', calibration)
        self.assertIn('confidence_coverage', calibration)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    unittest.main()
