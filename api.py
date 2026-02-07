"""
BudgetGuard FastAPI Service

REST API for cashflow forecasting and risk analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from budgetguard import (
    ExpenseCategory,
    BayesianExpenseModel,
    CashflowSimulator,
    ScenarioAnalyzer,
    BaselineComparator
)

app = FastAPI(
    title="BudgetGuard API",
    description="Cashflow forecasting and financial risk analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ExpenseCategoryInput(BaseModel):
    name: str
    mean: float = Field(gt=0, description="Average monthly spending")
    std: float = Field(gt=0, description="Standard deviation")
    alpha: float = Field(default=2.0, gt=0)
    beta: float = Field(default=2.0, gt=0)


class ForecastRequest(BaseModel):
    initial_balance: float
    monthly_income: float = 0.0
    categories: List[ExpenseCategoryInput]
    forecast_days: int = Field(ge=1, le=365)
    n_simulations: int = Field(default=10000, ge=1000, le=50000)
    savings_goal: Optional[float] = None


class ForecastResponse(BaseModel):
    days: int
    mean_balance: float
    percentile_5: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    overdraft_probability: float
    goal_miss_probability: float


class ScenarioRequest(BaseModel):
    initial_balance: float
    monthly_income: float = 0.0
    categories: List[ExpenseCategoryInput]
    forecast_days: int = Field(ge=1, le=365)
    category_caps: Dict[str, List[float]]


class ScenarioResult(BaseModel):
    scenario: str
    category: str
    cap: Optional[float]
    overdraft_prob: float
    mean_balance: float
    p5_balance: float
    risk_reduction: Optional[float] = None


class VarianceContribution(BaseModel):
    category: str
    variance: float
    std_dev: float
    mean: float
    variance_contribution_pct: float


class MultiHorizonRequest(BaseModel):
    initial_balance: float
    monthly_income: float = 0.0
    categories: List[ExpenseCategoryInput]
    horizons: List[int] = Field(default=[30, 60, 90])
    savings_goal: Optional[float] = None


# Helper Functions
def create_expense_model(categories: List[ExpenseCategoryInput]) -> BayesianExpenseModel:
    """Convert API input to expense model."""
    cat_dict = {}
    for cat in categories:
        cat_dict[cat.name] = ExpenseCategory(
            name=cat.name,
            mean=cat.mean,
            std=cat.std,
            alpha=cat.alpha,
            beta=cat.beta
        )
    return BayesianExpenseModel(cat_dict)


# API Endpoints
@app.get("/")
async def root():
    """API health check."""
    return {
        "service": "BudgetGuard API",
        "status": "operational",
        "version": "1.0.0"
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_cashflow(request: ForecastRequest):
    """
    Generate cashflow forecast with risk metrics.
    
    Returns distribution statistics and probability of overdraft/goal miss.
    """
    try:
        # Create model and simulator
        expense_model = create_expense_model(request.categories)
        simulator = CashflowSimulator(
            initial_balance=request.initial_balance,
            expense_model=expense_model,
            monthly_income=request.monthly_income
        )
        
        # Run simulation
        forecast = simulator.simulate(
            days=request.forecast_days,
            n_simulations=request.n_simulations,
            savings_goal=request.savings_goal
        )
        
        return ForecastResponse(
            days=forecast.days,
            mean_balance=forecast.mean_balance,
            percentile_5=forecast.percentile_5,
            percentile_25=forecast.percentile_25,
            percentile_50=forecast.percentile_50,
            percentile_75=forecast.percentile_75,
            percentile_95=forecast.percentile_95,
            overdraft_probability=forecast.overdraft_probability,
            goal_miss_probability=forecast.goal_miss_probability
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/forecast/multi-horizon")
async def multi_horizon_forecast(request: MultiHorizonRequest):
    """
    Generate forecasts for multiple time horizons (e.g., 30/60/90 days).
    """
    try:
        expense_model = create_expense_model(request.categories)
        simulator = CashflowSimulator(
            initial_balance=request.initial_balance,
            expense_model=expense_model,
            monthly_income=request.monthly_income
        )
        
        results = []
        for days in request.horizons:
            forecast = simulator.simulate(
                days=days,
                n_simulations=10000,
                savings_goal=request.savings_goal
            )
            
            results.append({
                "horizon_days": days,
                "mean_balance": forecast.mean_balance,
                "median_balance": forecast.percentile_50,
                "p5": forecast.percentile_5,
                "p95": forecast.percentile_95,
                "overdraft_probability": forecast.overdraft_probability,
                "goal_miss_probability": forecast.goal_miss_probability
            })
        
        return {"forecasts": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario-analysis", response_model=List[ScenarioResult])
async def analyze_scenarios(request: ScenarioRequest):
    """
    Analyze impact of spending caps on risk metrics.
    
    Returns risk reduction estimates for different category cap scenarios.
    """
    try:
        expense_model = create_expense_model(request.categories)
        simulator = CashflowSimulator(
            initial_balance=request.initial_balance,
            expense_model=expense_model,
            monthly_income=request.monthly_income
        )
        
        analyzer = ScenarioAnalyzer(simulator)
        results_df = analyzer.analyze_cap_impact(
            days=request.forecast_days,
            category_caps=request.category_caps
        )
        
        # Convert to response format
        results = []
        for _, row in results_df.iterrows():
            results.append(ScenarioResult(
                scenario=row['scenario'],
                category=row['category'],
                cap=row['cap'] if row['cap'] is not None else None,
                overdraft_prob=row['overdraft_prob'],
                mean_balance=row['mean_balance'],
                p5_balance=row['p5_balance'],
                risk_reduction=row.get('risk_reduction', None)
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/variance-analysis", response_model=List[VarianceContribution])
async def analyze_variance(request: ForecastRequest):
    """
    Identify primary contributors to cashflow variance.
    
    Returns variance decomposition showing which categories drive uncertainty.
    """
    try:
        expense_model = create_expense_model(request.categories)
        simulator = CashflowSimulator(
            initial_balance=request.initial_balance,
            expense_model=expense_model,
            monthly_income=request.monthly_income
        )
        
        analyzer = ScenarioAnalyzer(simulator)
        variance_df = analyzer.variance_decomposition(days=request.forecast_days)
        
        # Convert to response format
        results = []
        for _, row in variance_df.iterrows():
            results.append(VarianceContribution(
                category=row['category'],
                variance=row['variance'],
                std_dev=row['std_dev'],
                mean=row['mean'],
                variance_contribution_pct=row['variance_contribution_pct']
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline-comparison")
async def compare_baselines(request: ForecastRequest):
    """
    Compare Bayesian forecast against deterministic baselines.
    """
    try:
        expense_model = create_expense_model(request.categories)
        simulator = CashflowSimulator(
            initial_balance=request.initial_balance,
            expense_model=expense_model,
            monthly_income=request.monthly_income
        )
        
        # Bayesian forecast
        forecast = simulator.simulate(days=request.forecast_days)
        
        # Baseline comparisons
        ma_baseline = BaselineComparator.moving_average_forecast(
            [request.initial_balance]
        )
        rules_baseline = request.initial_balance + BaselineComparator.rules_based_budget(
            request.monthly_income
        ) * (request.forecast_days / 30)
        
        return {
            "bayesian_model": {
                "mean": forecast.mean_balance,
                "median": forecast.percentile_50,
                "confidence_interval": [forecast.percentile_5, forecast.percentile_95]
            },
            "moving_average": ma_baseline,
            "rules_based": rules_baseline,
            "forecast_days": request.forecast_days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
