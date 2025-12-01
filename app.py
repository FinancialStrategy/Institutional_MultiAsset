import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import scipy.stats as stats
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# Portfolio optimization imports
import pypfopt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import objective_functions
from pypfopt import CLA, plotting
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
from scipy.optimize import minimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# ============================================================
# ENHANCED CONFIGURATION WITH DYNAMIC PARAMETERS
# ============================================================

class EnhancedPortfolioConfig:
    """Extended configuration with dynamic parameters and validation"""

    def __init__(self, initial_capital: float, custom_assets: Dict[str, List[str]]):
        # Core parameters
        self.INITIAL_CAPITAL = initial_capital
        self.RISK_FREE_RATE = self._get_current_risk_free_rate()
        self.ANNUAL_TRADING_DAYS = 252

        # Enhanced asset universe with categorization
        self.ASSET_UNIVERSE = custom_assets
        
        # Flatten asset list with metadata
        self.ASSETS = []
        self.ASSET_CATEGORIES = {}
        for category, assets in self.ASSET_UNIVERSE.items():
            self.ASSETS.extend(assets)
            for asset in assets:
                self.ASSET_CATEGORIES[asset] = category

        # Benchmark - using ^GSPC (S&P 500 Index) or a regional index like XU100.
        self.BENCHMARK = '^GSPC' if 'US_Equities' in custom_assets else 'XU100.IS'

        # Dynamic date parameters
        self.END_DATE = datetime.now().strftime('%Y-%m-%d')
        self.START_DATE = (datetime.now() - timedelta(days=6*365)).strftime('%Y-%m-%d') # 6 years
        
        # Optimization parameters
        self.MIN_WEIGHT = 0.01
        self.MAX_WEIGHT = 0.20
        self.MAX_SECTOR_WEIGHT = 0.35

        # Advanced risk parameters
        self.VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

    def _get_current_risk_free_rate(self) -> float:
        """Dynamically fetch current risk-free rate"""
        try:
            # Use a high rate for a high-inflation economy like Turkey
            if self.BENCHMARK == 'XU100.IS':
                return 0.20 
            
            # Use 10-year Treasury yield (^TNX) as proxy for USD portfolio
            ticker = yf.Ticker('^TNX')
            hist = ticker.history(period='1d')
            if not hist.empty:
                current_yield = hist['Close'].iloc[-1] / 100
                return max(0.02, min(0.08, current_yield)) 
        except:
            pass
        return 0.0425 # Fallback to 4.25% for USD

# ============================================================
# COMPREHENSIVE HISTORICAL SCENARIOS
# ============================================================

class ComprehensiveHistoricalScenarios:
    SCENARIOS = {
        "COVID-19 Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
        "Q4 2018 Bear Market": ("2018-10-01", "2018-12-31"),
        "Russia-Ukraine War (Feb-Mar 2022)": ("2022-02-24", "2022-03-31"),
        "Lehman Collapse (Sep-Oct 2008)": ("2008-09-15", "2008-10-15"),
        "2022 Final Selloff (Sep-Oct 2022)": ("2022-09-01", "2022-10-31"),
    }
    
    def get_scenarios_in_range(self, start_date: str, end_date: str) -> Dict[str, Tuple[str, str]]:
        """Filter scenarios to include only those fully within the data range."""
        valid_scenarios = {}
        data_start = pd.to_datetime(start_date)
        data_end = pd.to_datetime(end_date)
        
        for name, (start, end) in self.SCENARIOS.items():
            scenario_start = pd.to_datetime(start)
            scenario_end = pd.to_datetime(end)
            
            if scenario_start >= data_start and scenario_end <= data_end:
                valid_scenarios[name] = (start, end)
        return valid_scenarios


# ============================================================
# ADVANCED DATA MANAGER
# ============================================================

class AdvancedDataManager:
    def __init__(self, config: EnhancedPortfolioConfig):
        self.config = config
        self.asset_prices = pd.DataFrame()
        self.asset_returns = pd.DataFrame()
        self.benchmark_returns = pd.Series()
        self.risk_free_rate = config.RISK_FREE_RATE
        self.log = []

    def _log(self, message: str):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    @st.cache_data(ttl=3600) # Cache data for one hour
    def load_data(_self) -> bool:
        """Load and validate all required data"""
        self = _self # use self for cached method
        self.log.clear()
        self._log("📥 Loading comprehensive market data...")

        try:
            successful_assets = []
            asset_dataframes = []

            for asset in self.config.ASSETS:
                try:
                    self._log(f"Downloading data for {asset}...")
                    ticker = yf.Ticker(asset)
                    data = ticker.history(
                        start=self.config.START_DATE,
                        end=self.config.END_DATE,
                        auto_adjust=True
                    )

                    if data.empty:
                        self._log(f"No data for {asset}")
                        continue

                    if 'Close' in data.columns:
                        close_prices = data['Close']
                        close_prices.index = close_prices.index.tz_localize(None)
                        close_prices = close_prices.rename(asset)
                        asset_dataframes.append(close_prices)
                        successful_assets.append(asset)
                        self._log(f"✅ Successfully loaded {asset}")
                    else:
                        self._log(f"No Close price for {asset}")

                except Exception as e:
                    self._log(f"Failed to download {asset}: {e}")
                    continue

            if len(successful_assets) < 5:
                self._log(f"Insufficient assets downloaded: {len(successful_assets)}")
                return False

            # Combine all asset prices
            self.asset_prices = pd.concat(asset_dataframes, axis=1)
            self.asset_prices = self.asset_prices.ffill().dropna()

            # Calculate returns
            self.asset_returns = self.asset_prices.pct_change().dropna()

            # Download benchmark data
            self._load_benchmark_data(successful_assets)

            # Align dates
            self._align_data_dates()

            self._log(f"✅ Data loaded: {len(self.asset_prices.columns)} assets, {len(self.asset_returns)} periods")
            return True

        except Exception as e:
            self._log(f"❌ Data loading failed: {e}")
            return False

    def _load_benchmark_data(self, successful_assets: list):
        """Load benchmark data with fallback logic"""
        try:
            self._log(f"Downloading benchmark data for {self.config.BENCHMARK}...")
            benchmark_ticker = yf.Ticker(self.config.BENCHMARK)
            benchmark_data = benchmark_ticker.history(
                start=self.config.START_DATE,
                end=self.config.END_DATE,
                auto_adjust=True
            )

            if not benchmark_data.empty and 'Close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['Close']
                benchmark_prices.index = benchmark_prices.index.tz_localize(None)
                self.benchmark_returns = benchmark_prices.pct_change().dropna()
                self._log(f"✅ Successfully loaded benchmark {self.config.BENCHMARK}")
            else:
                self._create_synthetic_benchmark(successful_assets)

        except Exception as e:
            self._log(f"Benchmark download failed: {e}")
            self._create_synthetic_benchmark(successful_assets)

    def _create_synthetic_benchmark(self, successful_assets: list):
        """Create synthetic benchmark"""
        if 'SPY' in successful_assets:
            self.benchmark_returns = self.asset_returns['SPY']
            self._log("Using SPY as fallback benchmark")
        else:
            self.benchmark_returns = self.asset_returns.mean(axis=1)
            self._log("Using equal-weighted synthetic benchmark")

    def _align_data_dates(self):
        """Align dates between datasets"""
        if self.asset_returns.empty or self.benchmark_returns.empty:
            return

        common_dates = self.asset_returns.index.intersection(self.benchmark_returns.index)

        if len(common_dates) > 0:
            self.asset_returns = self.asset_returns.loc[common_dates]
            self.benchmark_returns = self.benchmark_returns.loc[common_dates]
            self.asset_prices = self.asset_prices.loc[common_dates]
            self._log(f"✅ Aligned data to {len(common_dates)} common dates")

# ============================================================
# ENHANCED PORTFOLIO OPTIMIZER - COMPLETE IMPLEMENTATION
# ============================================================

class AdvancedPortfolioOptimizer:
    """Enhanced portfolio optimization with multiple methodologies and advanced features"""

    def __init__(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame,
                 risk_free_rate: float, config: EnhancedPortfolioConfig, log_func, asset_returns: pd.DataFrame = None):
        self.mu = expected_returns
        self.S = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.config = config
        self.asset_categories = config.ASSET_CATEGORIES
        self.asset_prices = None
        self.asset_returns = asset_returns # Store asset returns for momentum calculation
        self.log = log_func

    def set_asset_prices(self, asset_prices: pd.DataFrame):
        self.asset_prices = asset_prices

    def optimize_all_strategies(self) -> Dict[str, Dict]:
        """Optimize using multiple enhanced strategies with advanced techniques"""
        strategies = {}

        # Core optimization strategies (INCLUDING NEW ONES)
        optimization_methods = [
            ('Max Sharpe', self.max_sharpe_enhanced),
            ('Min Volatility', self.min_volatility_enhanced),
            ('Risk Parity', self.risk_parity_enhanced),
            ('Max Diversification', self.max_diversification_enhanced),
            ('Equal Risk Contribution', self.equal_risk_contribution),
            ('Black-Litterman', self.black_litterman_optimization),
            ('Robust Optimization', self.robust_optimization),
            ('Hierarchical Risk Parity', self.hierarchical_risk_parity),
            # --- NEW FACTOR/MOMENTUM STRATEGIES ---
            ('Momentum-Adj. Max Sharpe', self.momentum_adjusted_max_sharpe),
            ('Low-Vol Factor Tilt', self.factor_tilted_max_sharpe),
            # -------------------------------------
        ]

        for name, method in optimization_methods:
            try:
                weights = method()
                if weights is not None and self._validate_weights(weights):
                    strategies[name] = {
                        'weights': weights,
                        'description': self._get_strategy_description(name),
                        'optimization_method': name
                    }
                    self.log(f"✅ {name} optimization completed successfully")
                else:
                    self.log(f"⚠️ {name} optimization returned invalid weights")
            except Exception as e:
                self.log(f"⚠️ {name} optimization failed: {e}")

        strategies.update(self._add_benchmark_strategies())

        self.log(f"🎯 Successfully optimized {len(strategies)} portfolio strategies")
        return strategies

    # -------------------- NEW STRATEGY 1: MOMENTUM --------------------
    def momentum_adjusted_max_sharpe(self) -> Optional[Dict[str, float]]:
        """Max Sharpe using expected returns adjusted by 12-month momentum."""
        if self.asset_returns is None or len(self.asset_returns) < 252:
            self.log("Momentum calculation requires at least 1 year of data.")
            return self.max_sharpe_enhanced()
            
        try:
            # 1. Calculate 12-month (252 day) return, excluding the last month (21 days)
            lookback = 252
            skip = 21

            if len(self.asset_returns) < lookback + skip:
                 self.log(f"Not enough data for 12-month momentum ({lookback} days + {skip} skip days).")
                 return self.max_sharpe_enhanced()

            returns_for_mom = self.asset_returns.iloc[-(lookback + skip):-skip]
            momentum_returns = (1 + returns_for_mom).prod() - 1
            momentum_returns = momentum_returns.reindex(self.mu.index).fillna(0)
            
            # 2. Adjust expected returns: Blend historical return (70%) with momentum (30%)
            # We combine the existing mu with a tilt based on the momentum score
            
            # Simple scaling of expected returns based on momentum score (Annualized)
            mu_mom = self.mu * (1 + momentum_returns)
            
            # Normalize to current magnitude (keep the total level roughly the same)
            if mu_mom.mean() != 0:
                mu_mom = mu_mom * (self.mu.mean() / mu_mom.mean())
            else:
                mu_mom = self.mu

            # 3. Optimize using momentum-adjusted returns
            ef = EfficientFrontier(mu_mom, self.S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            return self._apply_weight_rounding(weights)
        
        except Exception as e:
            self.log(f"Momentum-Adj. optimization failed, falling back: {e}")
            return self.max_sharpe_enhanced()
    # ------------------------------------------------------------------

    # -------------------- NEW STRATEGY 2: FACTOR EXPOSURE --------------------
    def factor_tilted_max_sharpe(self) -> Optional[Dict[str, float]]:
        """Max Sharpe with a constraint on low volatility factor exposure (Factor Tilt)."""
        try:
            ef = EfficientFrontier(self.mu, self.S)
            
            # 1. Define Factor Exposure: Use Inverse Volatility as a proxy for the Low Vol Factor
            factor_exposure = 1 / np.sqrt(np.diag(self.S.values))
            factor_exposure_series = pd.Series(factor_exposure, index=self.mu.index)

            # 2. Calculate Factor Benchmark Exposure (e.g., Equal Weight Low Vol Exposure)
            factor_benchmark = factor_exposure_series.mean()
            
            # 3. Define the Factor Tilt Constraint (Weighted average factor exposure >= Benchmark)
            # We enforce a slight tilt to the Low Vol Factor (>= 101% of the average exposure)
            tilt_target = factor_benchmark * 1.01 
            
            # Since factor_exposure is a numpy array corresponding to self.mu.index order:
            def low_vol_tilt_constraint(w):
                # Constraint form: Portfolio_Exposure - Target >= 0
                portfolio_factor_exposure = w @ factor_exposure_series.values
                return portfolio_factor_exposure - tilt_target

            # 4. Apply constraints and optimize Max Sharpe
            self._apply_enhanced_constraints(ef)
            
            # pypfopt API: Add constraint using the custom function and type 'ineq' (for inequality)
            ef.add_constraint(low_vol_tilt_constraint, 'ineq')
            
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            # Verify constraint was met (optional, for logging)
            w_array = np.array([weights.get(t, 0.0) for t in self.mu.index])
            final_factor_exposure = w_array @ factor_exposure_series.values
            self.log(f"Factor Tilt: Target Low Vol Exp >= {tilt_target:.4f}, Actual: {final_factor_exposure:.4f}")
            
            return self._apply_weight_rounding(weights)

        except Exception as e:
            self.log(f"Factor Tilt optimization failed, falling back: {e}")
            return self.max_sharpe_enhanced()
    # ------------------------------------------------------------------------
    
    # --- EXISTING STRATEGIES START ---
    def max_sharpe_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()
            return self._apply_weight_rounding(weights)
        except Exception:
            return self.max_sharpe_fallback()

    def max_sharpe_fallback(self) -> Optional[Dict[str, float]]:
        try:
            cla = CLA(self.mu, self.S)
            cla.max_sharpe()
            weights = cla.clean_weights()
            return weights
        except Exception:
            return self.inverse_volatility()

    def min_volatility_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_objective(objective_functions.L2_reg, gamma=0.05)
            self._apply_enhanced_constraints(ef)
            ef.min_volatility()
            weights = ef.clean_weights()
            return self._apply_weight_rounding(weights)
        except Exception:
            return self.min_volatility_fallback()

    def min_volatility_fallback(self) -> Dict[str, float]:
        try:
            variances = np.diag(self.S.values)
            inverse_variance = 1 / np.maximum(variances, 1e-12)
            weights = inverse_variance / inverse_variance.sum()
            weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
            return self._apply_basic_constraints(weight_dict)
        except Exception:
            return self.equal_weight()

    def risk_parity_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            initial_weights = np.ones(len(volatilities)) / len(volatilities)
            weights = self._iterative_risk_parity(initial_weights)
            return self._apply_weight_rounding(weights)
        except Exception:
            return self.risk_parity_basic()

    def _iterative_risk_parity(self, initial_weights: np.ndarray, max_iter: int = 100) -> Dict[str, float]:
        weights = initial_weights.copy()
        n_assets = len(weights)
        
        for iteration in range(max_iter):
            portfolio_variance = weights @ self.S.values @ weights
            if portfolio_variance <= 0: break

            marginal_risk = self.S.values @ weights
            risk_contributions = weights * marginal_risk / portfolio_variance
            
            target_risk = 1.0 / n_assets
            risk_ratios = risk_contributions / target_risk
            
            adjustment = 1.0 / np.sqrt(np.maximum(risk_ratios, 1e-12))
            weights = weights * adjustment
            weights = weights / weights.sum()

            if np.max(np.abs(risk_ratios - 1.0)) < 0.01: break

        weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
        return self._apply_basic_constraints(weight_dict)

    def max_diversification_enhanced(self) -> Optional[Dict[str, float]]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))

            def diversification_ratio(w):
                weighted_avg_vol = w @ volatilities
                portfolio_vol = np.sqrt(w @ self.S.values @ w)
                return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0

            n_assets = len(self.mu)
            initial_weights = np.ones(n_assets) / n_assets
            bounds = [(self.config.MIN_WEIGHT, self.config.MAX_WEIGHT) for _ in range(n_assets)]

            def objective(weights):
                return -diversification_ratio(weights)

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            result = minimize(objective, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            if result.success:
                weights = {asset: float(w) for asset, w in zip(self.mu.index, result.x)}
                return self._apply_weight_rounding(weights)
            else:
                return self.max_diversification_basic()
        except Exception:
            return self.max_diversification_basic()

    def max_diversification_basic(self) -> Dict[str, float]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            diversification_weights = 1 / np.maximum(volatilities, 1e-12)
            weights = diversification_weights / diversification_weights.sum()
            weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
            return self._apply_basic_constraints(weight_dict)
        except Exception:
            return self.equal_weight()

    def equal_risk_contribution(self) -> Dict[str, float]:
        try:
            n_assets = len(self.mu)
            initial_weights = np.ones(n_assets) / n_assets

            def risk_contribution_variance(weights):
                w = np.array(weights)
                portfolio_variance = w @ self.S.values @ w
                if portfolio_variance <= 0: return 1.0
                marginal_risk = self.S.values @ w
                risk_contributions = w * marginal_risk / portfolio_variance
                return np.var(risk_contributions)

            bounds = [(self.config.MIN_WEIGHT, self.config.MAX_WEIGHT) for _ in range(n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

            result = minimize(risk_contribution_variance, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                weights = {asset: float(w) for asset, w in zip(self.mu.index, result.x)}
                return self._apply_weight_rounding(weights)
            else:
                return self.risk_parity_basic()
        except Exception:
            return self.risk_parity_basic()

    def black_litterman_optimization(self) -> Dict[str, float]:
        try:
            market_prior = market_implied_prior_returns(
                market_caps=None,
                cov_matrix=self.S,
                risk_aversion=1.0
            )

            bl = BlackLittermanModel(self.S, pi=market_prior)
            bl_returns = bl.bl_returns()
            
            ef = EfficientFrontier(bl_returns, self.S)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            return self._apply_weight_rounding(weights)
        except Exception:
            return self.max_sharpe_enhanced()

    def robust_optimization(self) -> Dict[str, float]:
        try:
            if self.asset_prices is not None:
                sample_cov = risk_models.sample_cov(self.asset_prices)
                ledoit_cov = risk_models.CovarianceShrinkage(self.asset_prices).ledoit_wolf()
                robust_cov = (sample_cov + ledoit_cov) / 2
                robust_cov = robust_cov.loc[self.mu.index, self.mu.index]
            else:
                robust_cov = self.S

            ef = EfficientFrontier(self.mu, robust_cov)
            self._apply_enhanced_constraints(ef)
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            weights = ef.clean_weights()

            return self._apply_weight_rounding(weights)
        except Exception:
            return self.max_sharpe_enhanced()

    def hierarchical_risk_parity(self) -> Dict[str, float]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            weights = np.ones(len(self.mu)) / len(self.mu)

            if len(self.mu) > 1 and self.asset_prices is not None:
                corr_matrix = risk_models.sample_cov(self.asset_prices).corr()
                corr_matrix = corr_matrix.loc[self.mu.index, self.mu.index].values
                
                dist = squareform(1 - corr_matrix)
                linkage_matrix = hierarchy.linkage(dist, method='single')
                clusters = hierarchy.fcluster(linkage_matrix, t=0.5, criterion='distance')
                
                for cluster_id in np.unique(clusters):
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    if len(cluster_indices) > 0:
                        cluster_vols = volatilities[cluster_indices]
                        cluster_weights = 1 / np.maximum(cluster_vols, 1e-12)
                        cluster_weights = cluster_weights / cluster_weights.sum()
                        weights[cluster_indices] = cluster_weights

                weights = weights / weights.sum()

            weight_dict = {asset: float(w) for asset, w in zip(self.mu.index, weights)}
            return self._apply_basic_constraints(weight_dict)
        except Exception as e:
            self.log(f"HRP optimization failed: {e}")
            return self.risk_parity_enhanced()

    def equal_weight(self) -> Dict[str, float]:
        n_assets = len(self.mu)
        weight = 1.0 / n_assets
        return {asset: weight for asset in self.mu.index}

    def inverse_volatility(self) -> Dict[str, float]:
        try:
            volatilities = np.sqrt(np.diag(self.S.values))
            inverse_vol = 1 / np.maximum(volatilities, 1e-12)
            weights = inverse_vol / inverse_vol.sum()
            weight_dict = {asset: float(weight) for asset, weight in zip(self.mu.index, weights)}
            return weight_dict
        except Exception:
            return self.equal_weight()
            
    def risk_parity_basic(self) -> Dict[str, float]:
        return self.inverse_volatility()

    def _apply_enhanced_constraints(self, ef: EfficientFrontier):
        ef.add_constraint(lambda w: w >= self.config.MIN_WEIGHT)
        ef.add_constraint(lambda w: w <= self.config.MAX_WEIGHT)

        if hasattr(self, 'asset_categories') and self.asset_categories:
            self._apply_sector_constraints(ef)

    def _apply_sector_constraints(self, ef: EfficientFrontier):
        try:
            categories = {}
            for asset, category in self.asset_categories.items():
                if asset in self.mu.index:
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(asset)
            # Sector constraint logic relies on pypfopt API. For this version, we rely on individual min/max bounds.
            pass
        except Exception as e:
            self.log(f"Sector constraint application failed: {e}")

    def _apply_basic_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        try:
            constrained_weights = {}
            total_weight = sum(weights.values())

            if total_weight <= 0: return self.equal_weight()

            min_weight = getattr(self.config, 'MIN_WEIGHT', 0.01)
            max_weight = getattr(self.config, 'MAX_WEIGHT', 0.20)

            for asset, weight in weights.items():
                normalized_weight = weight / total_weight
                constrained_weight = max(min_weight, min(max_weight, normalized_weight))
                constrained_weights[asset] = constrained_weight

            total_constrained = sum(constrained_weights.values())
            if total_constrained > 0:
                constrained_weights = {k: v/total_constrained for k, v in constrained_weights.items()}

            return constrained_weights
        except Exception:
            return self.equal_weight()

    def _apply_weight_rounding(self, weights: Dict[str, float]) -> Dict[str, float]:
        try:
            rounded_weights = {asset: round(weight, 4) for asset, weight in weights.items()}
            min_weight = getattr(self.config, 'MIN_WEIGHT', 0.01)
            cleaned_weights = {asset: weight for asset, weight in rounded_weights.items()
                               if weight >= min_weight}
            total = sum(cleaned_weights.values())
            if total > 0:
                final_weights = {asset: weight/total for asset, weight in cleaned_weights.items()}
                return final_weights
            else:
                return self.equal_weight()
        except Exception:
            return self.equal_weight()

    def _validate_weights(self, weights: Dict[str, float]) -> bool:
        if not weights: return False
        try:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01: return False
            if any(weight < -0.001 for weight in weights.values()): return False
            return True
        except Exception:
            return False

    def _get_strategy_description(self, strategy_name: str) -> str:
        descriptions = {
            'Max Sharpe': 'Maximum risk-adjusted returns (Sharpe ratio) with constraints',
            'Min Volatility': 'Minimum volatility portfolio with diversification objectives',
            'Risk Parity': 'Equal risk contribution across all assets using iterative methods',
            'Max Diversification': 'Maximum diversification ratio optimization',
            'Equal Risk Contribution': 'True equal risk contribution portfolio minimizing risk concentration',
            'Black-Litterman': 'Market equilibrium model incorporating investor views (equilibrium only here)',
            'Robust Optimization': 'Optimization using a blended (sample + shrinkage) covariance estimate',
            'Hierarchical Risk Parity': 'Clustering-based risk parity for improved diversification',
            'Momentum-Adj. Max Sharpe': 'Max Sharpe using expected returns adjusted by intermediate-term (12-1 month) momentum.',
            'Low-Vol Factor Tilt': 'Max Sharpe optimization subject to a minimum exposure constraint to the Low Volatility Factor.',
            'Equal Weight': 'Equal allocation across all assets (naive diversification benchmark)',
            'Inverse Volatility': 'Risk-based allocation weighted by inverse of volatility'
        }
        return descriptions.get(strategy_name, 'Advanced portfolio optimization strategy')

    def _add_benchmark_strategies(self) -> Dict[str, Dict]:
        return {
            'Equal Weight': {
                'weights': self.equal_weight(),
                'description': self._get_strategy_description('Equal Weight'),
                'optimization_method': 'Benchmark'
            },
            'Inverse Volatility': {
                'weights': self.inverse_volatility(),
                'description': self._get_strategy_description('Inverse Volatility'),
                'optimization_method': 'Benchmark'
            }
        }
# ============================================================
# COMPREHENSIVE METRICS CALCULATOR (Retained)
# ============================================================

class ComprehensiveMetricsCalculator:
    @staticmethod
    def calculate_performance_metrics(portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series,
                                      risk_free_rate: float) -> Dict[str, float]:
        if portfolio_returns.empty: return {}
        
        # Calculate returns
        total_return = (1 + portfolio_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        cumulative_return = total_return
        
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1

        # Risk-adjusted metrics
        excess_returns = portfolio_returns - risk_free_rate / 252
        std = portfolio_returns.std()
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / std if std > 0 else 0

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

        # Alpha and Beta
        covariance = portfolio_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        alpha = annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252) if len(active_returns) > 0 else 0
        information_ratio = (annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0


        return {
            'total_return': total_return, 'annual_return': annual_return, 
            'cumulative_return': cumulative_return, 'sharpe_ratio': sharpe_ratio, 
            'sortino_ratio': sortino_ratio, 'information_ratio': information_ratio,
            'alpha': alpha, 'beta': beta, 
            'tracking_error': tracking_error, 
        }

    @staticmethod
    def calculate_risk_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        if portfolio_returns.empty: return {}

        # Volatility & Drawdown
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # VaR and CVaR at multiple confidence levels
        var_90 = -np.percentile(portfolio_returns, 10)
        var_95 = -np.percentile(portfolio_returns, 5)
        var_99 = -np.percentile(portfolio_returns, 1)
        cvar_95 = -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        cvar_99 = -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean()

        # Higher moments
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()

        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        calmar_ratio = -annual_volatility / max_drawdown if max_drawdown < 0 else 0

        return {
            'annual_volatility': annual_volatility, 'max_drawdown': max_drawdown,
            'var_90': var_90, 'var_95': var_95, 'var_99': var_99,
            'cvar_95': cvar_95, 'cvar_99': cvar_99, 'skewness': skewness,
            'kurtosis': kurtosis, 'downside_std': downside_std,
            'calmar_ratio': calmar_ratio
        }

    @staticmethod
    def calculate_var_metrics(portfolio_returns: pd.Series, confidence_levels: List[float]) -> Dict[str, float]:
        var_metrics = {}
        for confidence in confidence_levels:
            var_key = f'var_{int(confidence*100)}'
            cvar_key = f'cvar_{int(confidence*100)}'
            var = -np.percentile(portfolio_returns, (1 - confidence) * 100)
            tail_returns = portfolio_returns[portfolio_returns <= -var]
            cvar = -tail_returns.mean() if len(tail_returns) > 0 else var
            var_metrics[var_key] = var
            var_metrics[cvar_key] = cvar
        return var_metrics

# ============================================================
# ENHANCED STRESS TEST CALCULATOR (Retained)
# ============================================================

class EnhancedStressTestCalculator:
    @staticmethod
    def calculate_stress_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                                 start_date: str, end_date: str) -> Dict[str, float]:
        try:
            start_dt = pd.to_datetime(start_date).tz_localize(None)
            end_dt = pd.to_datetime(end_date).tz_localize(None)

            portfolio_returns.index = portfolio_returns.index.tz_localize(None)
            benchmark_returns.index = benchmark_returns.index.tz_localize(None)

            mask = (portfolio_returns.index >= start_dt) & (portfolio_returns.index <= end_dt)
            stress_portfolio = portfolio_returns.loc[mask]
            stress_benchmark = benchmark_returns.loc[mask]

            if len(stress_portfolio) < 5: return {}

            portfolio_return = (1 + stress_portfolio).prod() - 1
            benchmark_return = (1 + stress_benchmark).prod() - 1
            relative_return = portfolio_return - benchmark_return

            portfolio_cumulative = (1 + stress_portfolio).cumprod()
            portfolio_peak = portfolio_cumulative.expanding().max()
            portfolio_drawdown = (portfolio_cumulative - portfolio_peak) / portfolio_peak
            max_drawdown = portfolio_drawdown.min()

            volatility = stress_portfolio.std() * np.sqrt(252)
            var_95 = -np.percentile(stress_portfolio, 5)
            cvar_95 = -stress_portfolio[stress_portfolio <= np.percentile(stress_portfolio, 5)].mean()

            covariance = stress_portfolio.cov(stress_benchmark)
            benchmark_variance = stress_benchmark.var()
            stress_beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0

            return {
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'relative_return': relative_return,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'stress_beta': stress_beta,
                'days_in_stress': len(stress_portfolio),
                'start_date': start_date,
                'end_date': end_date
            }

        except Exception:
            return {}
            
# ============================================================
# ADVANCED VISUALIZATION ENGINE WITH STREAMLIT OUTPUT
# ============================================================

class AdvancedVisualizationEngine:
    COLORS = {
        'performance': '#34495E', 'risk': '#1ABC9C', 'var': '#2980B9',
        'stress': '#9B59B6', 'strategy': '#E67E22', 'positive': '#27AE60',
        'negative': '#E74C3C', 'neutral': '#95A5A6'
    }

    def _format_dataframe(self, df: pd.DataFrame, header_color: str, percentage_cols: List[str], ratio_cols: List[str]) -> st.delta_generator.DeltaGenerator:
        """Formats a DataFrame for Streamlit display with conditional styling."""
        
        # Apply formatting
        format_dict = {}
        for col in percentage_cols:
            if col in df.columns: format_dict[col] = '{:+.2%}'
        for col in ratio_cols:
            if col in df.columns: format_dict[col] = '{:.3f}'
        
        df_styled = df.style.format(format_dict).set_table_styles([
            {'selector': 'th', 'props': [('background-color', header_color), ('color', 'white')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}
        ])
        
        return st.dataframe(df_styled, use_container_width=True)


    def create_comprehensive_performance_table(self, data: Dict[str, Dict[str, float]]) -> st.delta_generator.DeltaGenerator:
        df = pd.DataFrame(data).T
        performance_metrics = ['annual_return', 'cumulative_return', 'sharpe_ratio', 'sortino_ratio', 'information_ratio', 'alpha', 'beta', 'tracking_error']
        available_metrics = [m for m in performance_metrics if m in df.columns]
        df = df[available_metrics]
        
        display_names = {'annual_return': 'Annual Return', 'cumulative_return': 'Cumulative Return', 'sharpe_ratio': 'Sharpe Ratio', 'sortino_ratio': 'Sortino Ratio', 'information_ratio': 'Info Ratio', 'alpha': 'Alpha', 'beta': 'Beta', 'tracking_error': 'Tracking Error'}
        df = df.rename(columns=display_names)
        
        percentage_cols = ['Annual Return', 'Cumulative Return', 'Tracking Error', 'Alpha']
        ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Info Ratio', 'Beta']
        
        return self._format_dataframe(df, self.COLORS['performance'], percentage_cols, ratio_cols)

    def create_comprehensive_risk_table(self, data: Dict[str, Dict[str, float]]) -> st.delta_generator.DeltaGenerator:
        df = pd.DataFrame(data).T
        risk_metrics = ['annual_volatility', 'downside_std', 'max_drawdown', 'skewness', 'kurtosis', 'var_95', 'cvar_95', 'calmar_ratio']
        available_metrics = [m for m in risk_metrics if m in df.columns]
        df = df[available_metrics]
        
        display_names = {'annual_volatility': 'Annual Vol', 'downside_std': 'Downside Vol', 'max_drawdown': 'Max Drawdown', 'skewness': 'Skewness', 'kurtosis': 'Kurtosis', 'var_95': 'VaR 95%', 'cvar_95': 'CVaR 95%', 'calmar_ratio': 'Calmar Ratio'}
        df = df.rename(columns=display_names)
        
        percentage_cols = ['Annual Vol', 'Downside Vol', 'Max Drawdown', 'VaR 95%', 'CVaR 95%']
        ratio_cols = ['Skewness', 'Kurtosis', 'Calmar Ratio']
        
        return self._format_dataframe(df, self.COLORS['risk'], percentage_cols, ratio_cols)

    def create_advanced_var_table(self, data: Dict[str, Dict[str, float]], confidence_levels: List[float]) -> st.delta_generator.DeltaGenerator:
        df = pd.DataFrame(data).T
        var_metrics = [f'var_{int(conf*100)}' for conf in confidence_levels] + [f'cvar_{int(conf*100)}' for conf in confidence_levels]
        available_metrics = [m for m in var_metrics if m in df.columns]
        df = df[available_metrics]
        
        display_names = {f'var_{int(c*100)}': f'VaR {int(c*100)}%' for c in confidence_levels}
        display_names.update({f'cvar_{int(c*100)}': f'CVaR {int(c*100)}%' for c in confidence_levels})
        df = df.rename(columns=display_names)
        
        percentage_cols = df.columns.tolist()
        
        return self._format_dataframe(df, self.COLORS['var'], percentage_cols, [])

    def create_historical_stress_analysis_table(self, stress_results: Dict[str, Dict[str, Any]]) -> st.delta_generator.DeltaGenerator:
        if not stress_results:
            st.info("No stress test data available within the analysis period.")
            return

        df = pd.DataFrame(stress_results).T
        stress_metrics = ['portfolio_return', 'benchmark_return', 'relative_return', 'max_drawdown', 'volatility', 'var_95', 'cvar_95', 'stress_beta', 'days_in_stress']
        available_metrics = [m for m in stress_metrics if m in df.columns]
        df = df[available_metrics]

        display_names = {
            'portfolio_return': 'Port. Ret', 'benchmark_return': 'Bench. Ret', 'relative_return': 'Relative Ret',
            'max_drawdown': 'Max DD', 'volatility': 'Stress Vol', 'var_95': 'VaR 95%', 
            'cvar_95': 'CVaR 95%', 'stress_beta': 'Stress Beta', 'days_in_stress': 'Duration Days'
        }
        df = df.rename(columns=display_names)

        percentage_cols = ['Port. Ret', 'Bench. Ret', 'Relative Ret', 'Max DD', 'Stress Vol', 'VaR 95%', 'CVaR 95%']
        ratio_cols = ['Stress Beta', 'Duration Days']

        return self._format_dataframe(df, self.COLORS['stress'], percentage_cols, ratio_cols)

    def create_advanced_efficient_frontier(self, mu: pd.Series, S: pd.DataFrame, strategies: Dict[str, Dict], risk_free_rate: float) -> go.Figure:
        try:
            # --- Robust Input Checks ---
            mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
            S = S.loc[mu.index, mu.index]
            if len(mu) < 2: raise ValueError("Fewer than two assets after sanitization.")

            # --- Calculate Frontier and Key Points ---
            ef = EfficientFrontier(mu, S)
            min_vol_ret, min_vol_vol = ef.min_volatility().portfolio_performance()[::2]
            
            ef_ms = EfficientFrontier(mu, S)
            max_sharpe_ret, max_sharpe_vol = ef_ms.max_sharpe(risk_free_rate=risk_free_rate).portfolio_performance()[::2]
            
            # Use discrete points for robust plotting
            target_returns = np.linspace(min_vol_ret, max_sharpe_ret * 1.5, 60)
            efficient_vols, efficient_rets = [], []
            for tr in target_returns:
                try:
                    ef_p = EfficientFrontier(mu, S)
                    ef_p.efficient_return(target_return=tr)
                    perf = ef_p.portfolio_performance()
                    efficient_rets.append(perf[0])
                    efficient_vols.append(perf[1])
                except Exception:
                    continue

            fig = go.Figure()

            # 1. Efficient Frontier Line
            if efficient_vols:
                fig.add_trace(go.Scatter(
                    x=efficient_vols, y=efficient_rets, mode='lines', name='Efficient Frontier',
                    line=dict(color=self.COLORS['performance'], width=3),
                    hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))

            # 2. Individual Assets
            individual_volatilities = np.sqrt(np.diag(S.values))
            fig.add_trace(go.Scatter(
                x=individual_volatilities, y=mu.values, mode='markers', name='Individual Assets',
                marker=dict(size=10, color=self.COLORS['risk'], symbol='circle'),
                text=list(mu.index),
                hovertemplate='<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            # 3. Strategy Points
            for strategy_name, strategy_info in strategies.items():
                weights = strategy_info.get('weights', {})
                if weights:
                    w = np.array([weights.get(t, 0.0) for t in mu.index], dtype=float)
                    w = w / np.sum(w) if np.sum(w) > 0 else w 
                    portfolio_return = float(np.dot(w, mu.values))
                    portfolio_vol = float(np.sqrt(w @ S.values @ w))

                    fig.add_trace(go.Scatter(
                        x=[portfolio_vol], y=[portfolio_return], mode='markers', name=strategy_name,
                        marker=dict(size=16, symbol='star', color=self.COLORS['strategy'], line=dict(width=2, color='white')),
                        hovertemplate=f'<b>{strategy_name}</b><br>Risk: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>'
                    ))

            # 4. Capital Market Line
            if max_sharpe_vol and max_sharpe_vol > 0:
                sharpe_ratio = (max_sharpe_ret - risk_free_rate) / max_sharpe_vol
                x_max = max(efficient_vols + individual_volatilities.tolist()) * 1.25 if efficient_vols else 0.5
                x_range = np.linspace(0, x_max, 50)
                cml_returns = risk_free_rate + sharpe_ratio * x_range
                fig.add_trace(go.Scatter(
                    x=x_range, y=cml_returns, mode='lines', name='Capital Market Line',
                    line=dict(color=self.COLORS['var'], dash='dash', width=2),
                    hoverinfo='skip'
                ))

            fig.update_layout(
                title_text='📈 Advanced Efficient Frontier Analysis',
                xaxis_title="Annual Volatility (Risk)", yaxis_title="Annual Return",
                hovermode='closest', showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                width=None, height=650, template="plotly_white",
                xaxis=dict(tickformat='.0%'), yaxis=dict(tickformat='.0%')
            )
            return fig

        except Exception as e:
            st.error(f"Efficient frontier plotting failed: {e}")
            return go.Figure().add_annotation(text="Efficient frontier not available", x=0.5, y=0.5, showarrow=False)

    def create_weight_allocation_chart(self, strategies: Dict[str, Dict]) -> go.Figure:
        if not strategies:
            return go.Figure().add_annotation(text="No strategies available", x=0.5, y=0.5, showarrow=False)

        all_assets = sorted(list(set(a for info in strategies.values() for a in info.get('weights', {}).keys())))
        
        data = []
        for strategy_name, strategy_info in strategies.items():
            weights = strategy_info.get('weights', {})
            weight_values = [weights.get(asset, 0) for asset in all_assets]

            data.append(go.Bar(
                name=strategy_name, x=all_assets, y=weight_values,
                hovertemplate='<b>%{x}</b><br>Strategy: ' + strategy_name + '<br>Weight: %{y:.2%}<extra></extra>'
            ))

        fig = go.Figure(data=data)
        fig.update_layout(
            title_text='📊 Portfolio Allocation by Strategy', barmode='group',
            xaxis=dict(title="Assets"),
            yaxis=dict(title="Weight Allocation", tickformat='.0%'),
            width=None, height=600, template="plotly_white"
        )
        return fig

# ============================================================
# MAIN PORTFOLIO ENGINE
# ============================================================

class QFAPortfolioEngine:
    def __init__(self, config: EnhancedPortfolioConfig, log_func):
        self.config = config
        self.log_func = log_func
        self.data_manager = AdvancedDataManager(self.config)
        self.visualizer = AdvancedVisualizationEngine()
        self.metrics_calculator = ComprehensiveMetricsCalculator()
        self.stress_calculator = EnhancedStressTestCalculator()

        self.asset_prices = None
        self.asset_returns = None
        self.benchmark_returns = None
        self.risk_free_rate = None

        self.strategies = {}
        self.comprehensive_performance_data = {}
        self.comprehensive_risk_data = {}
        self.var_data = {}
        self.stress_test_results = {}

    @staticmethod
    def _ensure_psd(S: pd.DataFrame, min_eig: float = 1e-10) -> pd.DataFrame:
        try:
            A = S.values
            A = 0.5 * (A + A.T)
            vals, vecs = np.linalg.eigh(A)
            vals_clipped = np.clip(vals, min_eig, None)
            A_psd = (vecs @ np.diag(vals_clipped) @ vecs.T)
            A_psd = 0.5 * (A_psd + A_psd.T)
            return pd.DataFrame(A_psd, index=S.index, columns=S.columns)
        except Exception:
            return S.copy()

    def _compute_sanitized_mu_cov(self) -> Tuple[pd.Series, pd.DataFrame]:
        try:
            mu = expected_returns.mean_historical_return(self.asset_prices)
        except Exception:
            mu = self.asset_returns.mean() * self.config.ANNUAL_TRADING_DAYS

        try:
            S = risk_models.CovarianceShrinkage(self.asset_prices).ledoit_wolf()
        except Exception:
            S = risk_models.sample_cov(self.asset_prices)

        mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
        S = S.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').dropna(axis=1, how='any')

        keep = mu.index.intersection(S.index)
        mu = mu.loc[keep]
        S = S.loc[keep, keep]

        diag = np.diag(S.values)
        valid = diag > 1e-12
        if not np.all(valid):
            mu = mu.iloc[valid]
            S = S.iloc[valid, valid]

        S = self._ensure_psd(S)

        if len(mu) < 2:
            raise ValueError("Insufficient valid assets after sanitization (need at least 2).")

        self.log_func(f"Sanitized asset set for EF: {list(mu.index)}")
        return mu, S

    def run_analysis(self):
        self.log_func("🚀 Starting Comprehensive Portfolio Analysis...")
        
        try:
            if not self.data_manager.load_data():
                st.error("Data loading failed. Please check asset tickers and try again.")
                return

            self.asset_prices = self.data_manager.asset_prices
            self.asset_returns = self.data_manager.asset_returns
            self.benchmark_returns = self.data_manager.benchmark_returns
            self.risk_free_rate = self.data_manager.risk_free_rate

            mu, S = self._compute_sanitized_mu_cov()

            # --- MODIFIED: Pass asset_returns to the optimizer for momentum strategy ---
            optimizer = AdvancedPortfolioOptimizer(
                mu, S, self.risk_free_rate, self.config, self.log_func, 
                asset_returns=self.asset_returns 
            )
            # -------------------------------------------------------------------------
            
            optimizer.set_asset_prices(self.asset_prices)
            self.strategies = optimizer.optimize_all_strategies()

            self._calculate_comprehensive_metrics()
            self._run_enhanced_stress_tests()
            
            self.log_func("✅ Comprehensive Portfolio Analysis Complete!")

        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            self.log_func(f"❌ Analysis failed: {e}")

    def _get_portfolio_returns(self, weights: Dict[str, float]) -> pd.Series:
        portfolio_returns = pd.Series(0.0, index=self.asset_returns.index)
        aligned_assets = [a for a in weights.keys() if a in self.asset_returns.columns and weights[a] > 0]
        if not aligned_assets: return portfolio_returns.dropna()

        w_vals = np.array([weights[a] for a in aligned_assets], dtype=float)
        s = w_vals.sum()
        if s <= 0: return portfolio_returns.dropna()
        w_vals /= s

        for asset, weight in zip(aligned_assets, w_vals):
            portfolio_returns += self.asset_returns[asset] * weight

        return portfolio_returns.dropna()

    def _run_enhanced_stress_tests(self):
        self.log_func("📉 Running enhanced historical stress tests...")
        self.stress_test_results = {}
        if not self.strategies: return

        primary_strategy = list(self.strategies.keys())[0]
        portfolio_returns = self._get_portfolio_returns(self.strategies[primary_strategy]['weights'])
        
        if portfolio_returns.empty: return

        available_scenarios = ComprehensiveHistoricalScenarios().get_scenarios_in_range(
            self.asset_returns.index.min().strftime('%Y-%m-%d'),
            self.asset_returns.index.max().strftime('%Y-%m-%d')
        )

        for scenario_name, (start_date, end_date) in available_scenarios.items():
            stress_metrics = self.stress_calculator.calculate_stress_metrics(
                portfolio_returns, self.benchmark_returns, start_date, end_date
            )
            if stress_metrics:
                self.stress_test_results[scenario_name] = stress_metrics
                self.log_func(f"✅ Stress test completed: {scenario_name}")

    def _calculate_comprehensive_metrics(self):
        self.log_func("📊 Calculating comprehensive metrics...")
        self.comprehensive_performance_data = {}
        self.comprehensive_risk_data = {}
        self.var_data = {}

        for strategy_name, strategy_info in self.strategies.items():
            portfolio_returns = self._get_portfolio_returns(strategy_info['weights'])

            if portfolio_returns.empty: continue

            performance_metrics = self.metrics_calculator.calculate_performance_metrics(
                portfolio_returns, self.benchmark_returns, self.risk_free_rate
            )
            risk_metrics = self.metrics_calculator.calculate_risk_metrics(
                portfolio_returns, self.benchmark_returns
            )
            var_metrics = self.metrics_calculator.calculate_var_metrics(
                portfolio_returns, self.config.VAR_CONFIDENCE_LEVELS
            )

            self.comprehensive_performance_data[strategy_name] = performance_metrics
            self.comprehensive_risk_data[strategy_name] = risk_metrics
            self.var_data[strategy_name] = var_metrics

        if not self.benchmark_returns.empty:
            benchmark_performance = self.metrics_calculator.calculate_performance_metrics(
                self.benchmark_returns, self.benchmark_returns, self.risk_free_rate
            )
            benchmark_risk = self.metrics_calculator.calculate_risk_metrics(
                self.benchmark_returns, self.benchmark_returns
            )
            benchmark_var = self.metrics_calculator.calculate_var_metrics(
                self.benchmark_returns, self.config.VAR_CONFIDENCE_LEVELS
            )

            self.comprehensive_performance_data['Benchmark'] = benchmark_performance
            self.comprehensive_risk_data['Benchmark'] = benchmark_risk
            self.var_data['Benchmark'] = benchmark_var

    def get_optimal_strategy(self, criteria: str = "sharpe") -> str:
        """Get optimal strategy based on specified criteria"""
        if not self.comprehensive_performance_data: return ""

        best_strategy = ""
        best_value = -float('inf')

        for strategy, metrics in self.comprehensive_performance_data.items():
            if strategy == 'Benchmark': continue

            # Prioritize Max Sharpe first, then check other metrics
            value = metrics.get('sharpe_ratio', -float('inf'))
            if criteria == "return":
                value = metrics.get('annual_return', -float('inf'))
            elif criteria == "risk_adjusted":
                value = metrics.get('sortino_ratio', -float('inf'))
            
            if value > best_value:
                best_value = value
                best_strategy = strategy

        return best_strategy

# ============================================================
# STREAMLIT UI INTEGRATION
# ============================================================

def get_default_global_assets():
    return {
        'US_Equities': ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA'],
        'International_Equities': ['VEA', 'VWO', 'EWJ', 'EZU'],
        'Fixed_Income': ['BND', 'TLT', 'IEF', 'LQD', 'MUB', 'HYG'],
        'Real_Assets': ['GLD', 'SLV', 'VNQ', 'GSG', 'DBB'],
        'Sectors': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU']
    }

def get_turkish_equities():
    # Proxy list for 50 most active Borsa Istanbul stocks (using common tickers with .IS suffix)
    return {
        'Turkish_Equities': [
            'GARAN.IS', 'HALKB.IS', 'AKBNK.IS', 'THYAO.IS', 'EREGL.IS', 'TUPRS.IS', 'SAHOL.IS', 
            'KOZAA.IS', 'TCELL.IS', 'PETKM.IS', 'ISCTR.IS', 'ASELS.IS', 'KCHOL.IS', 'BIMAS.IS', 
            'GUBRF.IS', 'ODAS.IS', 'ARCLK.IS', 'VESTL.IS', 'SASA.IS', 'PGSUS.IS', 'TKFEN.IS', 
            'MGROS.IS', 'YATAS.IS', 'DOAS.IS', 'FROTO.IS', 'TOASO.IS', 'TATEN.IS', 'ENJSA.IS',
            'CIMSA.IS', 'AYDEM.IS', 'MAVI.IS', 'SEKFK.IS', 'ALARK.IS', 'DEVA.IS', 'SISE.IS',
            'KOZAL.IS', 'TAVHL.IS', 'ULKER.IS', 'KARTN.IS', 'LOGO.IS', 'IZMDC.IS',
            'EGEEN.IS', 'OTKAR.IS', 'ANHYT.IS', 'TRGYO.IS', 'AVISA.IS', 'TSKB.IS', 'GSDHO.IS',
            'AKSA.IS', 'YKBNK.IS', 'DENGE.IS' # Added two more to ensure 50
        ]
    }

def run_streamlit_tab(tab_title: str, config: EnhancedPortfolioConfig):
    
    st.header(f"🏛️ {tab_title}")
    
    # Simple logging display
    log_placeholder = st.empty()
    
    # Simple logger function to update the placeholder
    def update_log(message):
        if 'log_history' not in st.session_state:
            st.session_state['log_history'] = []
        st.session_state['log_history'].append(message)
        log_placeholder.code('\n'.join(st.session_state['log_history'][-10:]), language='text')

    if 'log_history' in st.session_state:
        del st.session_state['log_history'] # Clear log for fresh run

    engine = QFAPortfolioEngine(config, update_log)

    st.sidebar.markdown(f"### {tab_title} Configuration")
    st.sidebar.markdown(f"**Initial Capital:** ${config.INITIAL_CAPITAL:,.0f}")
    st.sidebar.markdown(f"**Risk-Free Rate:** {config.RISK_FREE_RATE:.2%}")
    
    if st.button(f"✨ Run {tab_title} Analysis", key=f"run_button_{tab_title}"):
        with st.spinner("Analyzing portfolio, please wait..."):
            engine.run_analysis()

    if engine.strategies:
        st.success("Analysis complete!")
        
        optimal_sharpe = engine.get_optimal_strategy("sharpe")
        
        st.markdown("---")
        
        st.subheader("🎯 Optimal Strategy Insight (Max Sharpe)")
        st.info(f"The **Maximum Sharpe Ratio** strategy is: **{optimal_sharpe}**.")
        if optimal_sharpe:
            sharpe_value = engine.comprehensive_performance_data[optimal_sharpe].get('sharpe_ratio', 0)
            st.markdown(f"**Sharpe Ratio:** `{sharpe_value:.3f}`")

        # --- Tabbed Results ---
        tab_eff, tab_weights, tab_perf, tab_risk, tab_stress = st.tabs([
            "Efficient Frontier", "Asset Allocation", "Performance Metrics", "Risk Metrics (VaR/CVaR)", "Stress Testing"
        ])
        
        with tab_eff:
            st.subheader("📈 Advanced Efficient Frontier Analysis")
            mu, S = engine._compute_sanitized_mu_cov()
            fig_eff = engine.visualizer.create_advanced_efficient_frontier(mu, S, engine.strategies, engine.risk_free_rate)
            st.plotly_chart(fig_eff, use_container_width=True)
            
            st.subheader("Strategy Characteristics Comparison")
            # Create strategy comparison table
            strategy_data = {}
            for strategy_name, strategy_info in engine.strategies.items():
                weights = strategy_info.get('weights', {})
                total_assets = len([w for w in weights.values() if w > 0.001])
                max_weight = max(weights.values()) if weights else 0
                concentration = sum(w**2 for w in weights.values())
                
                strategy_data[strategy_name] = {
                    'Description': strategy_info.get('description', ''),
                    'Assets Count': total_assets,
                    'Max Weight': f"{max_weight:.1%}",
                    'Concentration Index (HHI)': f"{concentration:.3f}",
                    'Diversification Score': f"{1 - concentration:.3f}"
                }
            st.dataframe(pd.DataFrame(strategy_data).T, use_container_width=True)

        with tab_weights:
            st.subheader("📊 Portfolio Weight Allocation by Strategy")
            fig_weights = engine.visualizer.create_weight_allocation_chart(engine.strategies)
            st.plotly_chart(fig_weights, use_container_width=True)
            
        with tab_perf:
            st.subheader("📈 Performance Metrics Comparison")
            engine.visualizer.create_comprehensive_performance_table(engine.comprehensive_performance_data)

        with tab_risk:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⚠️ Core Risk Metrics")
                engine.visualizer.create_comprehensive_risk_table(engine.comprehensive_risk_data)
            with col2:
                st.subheader("🛡️ Value at Risk (VaR) & CVaR")
                engine.visualizer.create_advanced_var_table(engine.var_data, config.VAR_CONFIDENCE_LEVELS)

        with tab_stress:
            st.subheader("🌪️ Historical Stress Testing Results")
            engine.visualizer.create_historical_stress_analysis_table(engine.stress_test_results)
            
# ============================================================
# MAIN STREAMLIT APP
# ============================================================

def main_app():
    st.set_page_config(layout="wide", page_title="QFA Institutional Portfolio Optimization")
    
    st.title("🏆 QFA Institutional Portfolio Optimization")
    st.markdown("### Advanced Optimization and Risk Analysis for Institutional Mandates")
    st.markdown("---")

    # --- Global Configuration Sidebar ---
    st.sidebar.title("Global Configuration")
    
    # USD Portfolio Config
    usd_capital = st.sidebar.number_input(
        "USD Portfolio Capital ($):",
        min_value=1_000_000,
        max_value=1_000_000_000,
        value=100_000_000,
        step=5_000_000,
        key="usd_cap"
    )
    
    # TRY Portfolio Config
    try_capital = st.sidebar.number_input(
        "TRY Portfolio Capital ($):",
        min_value=1_000_000,
        max_value=100_000_000,
        value=10_000_000,
        step=1_000_000,
        key="try_cap"
    )

    # --- Main Tabs ---
    tab_usd, tab_try = st.tabs([
        "🌍 Global Mandate (USD)", 
        "🇹🇷 Turkish Equities Mandate (TRY)"
    ])

    with tab_usd:
        usd_config = EnhancedPortfolioConfig(usd_capital, get_default_global_assets())
        run_streamlit_tab("Global Mandate (USD)", usd_config)

    with tab_try:
        try_config = EnhancedPortfolioConfig(try_capital, get_turkish_equities())
        run_streamlit_tab("Turkish Equities Mandate (TRY)", try_config)

if __name__ == "__main__":
    main_app()