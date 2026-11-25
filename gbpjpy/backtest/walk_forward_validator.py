"""
GBPJPY Walk-Forward & Anti-Overfitting Validation Engine (Milestone 5)
Provides:
- Strict chronological In-Sample / Out-of-Sample dataset partitioning
- 24-fold rolling walk-forward window generation with zero lookahead data leakage
- Walk-Forward Efficiency (WFE) ratio calculation with >=60% threshold governance
- Monte Carlo trade sequence permutation and drawdown distribution modeling
- Parameter stability and sensitivity analysis across neighborhood grids
"""

from datetime import datetime, timezone, timedelta
import logging
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Sequence
import numpy as np
import pandas as pd

try:
    from gbpjpy.backtest.engine import BacktestEngine, BacktestResult
    from gbpjpy.backtest.metrics import calculate_metrics, PerformanceMetrics
    from gbpjpy.strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
    from gbpjpy.core.data_engine import DataEngine
except ImportError:
    try:
        from .engine import BacktestEngine, BacktestResult
        from .metrics import calculate_metrics, PerformanceMetrics
        from ..strategies.gbpjpy_scalper import GBPJPYDualRegimeScalper
        from ..core.data_engine import DataEngine
    except ImportError:
        BacktestEngine = None
        BacktestResult = None
        calculate_metrics = None
        PerformanceMetrics = None
        GBPJPYDualRegimeScalper = None
        DataEngine = None

logger = logging.getLogger("GBPJPY.WalkForwardValidator")


class WalkForwardWindow(dict):
    """
    Container for a single walk-forward train/test temporal window.
    Supports dual attribute and dictionary indexing.
    """
    def __init__(
        self,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        fold: int = 1,
        fold_idx: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        f_idx = int(fold if fold_idx is None else fold_idx)
        data = {
            "fold": f_idx,
            "fold_idx": f_idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "metadata": metadata or {},
        }
        data.update(kwargs)
        for k, v in data.items():
            self[k] = v

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'WalkForwardWindow' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# Alias WindowRecord
WindowRecord = WalkForwardWindow


class MonteCarloResult(dict):
    """
    Container for Monte Carlo trade permutation simulation results.
    Inherits from dict and supports dual attribute and dictionary indexing.
    """
    _ALIASES = {
        "drawdown_95th": "p95_drawdown",
        "p95_drawdown": "drawdown_95th",
        "max_drawdown_95": "p95_drawdown",
        "drawdown_99th": "p99_drawdown",
        "p99_drawdown": "drawdown_99th",
        "max_drawdown_99": "p99_drawdown",
        "probability_of_ruin": "ruin_probability",
        "ruin_probability": "probability_of_ruin",
        "risk_of_ruin": "ruin_probability",
    }

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        mean_terminal_equity: float = 250.0,
        worst_drawdown: float = 0.0,
        median_drawdown: float = 0.0,
        p95_drawdown: float = 0.0,
        p99_drawdown: float = 0.0,
        ruin_probability: float = 0.0,
        iterations: int = 500,
        initial_balance: float = 250.0,
        terminal_equities: Optional[List[float]] = None,
        drawdowns: Optional[List[float]] = None,
        **kwargs
    ):
        super().__init__()
        initial = dict(data or {})
        initial.update(kwargs)
        if "mean_terminal_equity" not in initial:
            initial["mean_terminal_equity"] = round(float(mean_terminal_equity), 4)
        if "worst_drawdown" not in initial:
            initial["worst_drawdown"] = round(float(worst_drawdown), 4)
        if "median_drawdown" not in initial:
            initial["median_drawdown"] = round(float(median_drawdown), 4)
        if "p95_drawdown" not in initial:
            initial["p95_drawdown"] = round(float(p95_drawdown), 4)
        if "p99_drawdown" not in initial:
            initial["p99_drawdown"] = round(float(p99_drawdown), 4)
        if "ruin_probability" not in initial:
            initial["ruin_probability"] = round(float(ruin_probability), 4)
        if "iterations" not in initial:
            initial["iterations"] = int(iterations)
        if "initial_balance" not in initial:
            initial["initial_balance"] = round(float(initial_balance), 2)
        if "terminal_equities" not in initial:
            initial["terminal_equities"] = terminal_equities or []
        if "drawdowns" not in initial:
            initial["drawdowns"] = drawdowns or []

        # Set default aliases
        initial.setdefault("drawdown_95th", initial.get("p95_drawdown", 0.0))
        initial.setdefault("max_drawdown_95", initial.get("p95_drawdown", 0.0))
        initial.setdefault("drawdown_99th", initial.get("p99_drawdown", 0.0))
        initial.setdefault("max_drawdown_99", initial.get("p99_drawdown", 0.0))
        initial.setdefault("probability_of_ruin", initial.get("ruin_probability", 0.0))
        initial.setdefault("risk_of_ruin", initial.get("ruin_probability", 0.0))
        initial.setdefault("median_terminal_equity", initial.get("mean_terminal_equity", 0.0))
        initial.setdefault("worst_drawdown_pct", (initial.get("worst_drawdown", 0.0) / max(1.0, initial.get("initial_balance", 250.0))) * 100.0)
        initial.setdefault("mean_drawdown", initial.get("median_drawdown", 0.0))

        for k, v in initial.items():
            self[k] = v

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        alt = self._ALIASES.get(name)
        if alt and alt in self:
            return self[alt]
        raise AttributeError(f"'MonteCarloResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __getitem__(self, key: str) -> Any:
        if key in self:
            return super().__getitem__(key)
        alt = self._ALIASES.get(key)
        if alt and alt in self:
            return super().__getitem__(alt)
        return super().__getitem__(key)


class WalkForwardResult(dict):
    """
    Container for complete Walk-Forward Analysis and Anti-Overfitting validation results.
    Inherits from dict and supports dual attribute and dictionary indexing.
    """
    def __init__(
        self,
        wfe: float = 0.0,
        passed: bool = False,
        folds_total: int = 0,
        folds_passed: int = 0,
        mean_is_return: float = 0.0,
        mean_oos_return: float = 0.0,
        fold_details: Optional[List[Dict[str, Any]]] = None,
        folds: Optional[Any] = None,
        is_metrics: Optional[Any] = None,
        oos_metrics: Optional[Any] = None,
        monte_carlo: Optional[Any] = None,
        parameter_stability: float = 0.0,
        is_robust: Optional[bool] = None,
        wfe_threshold: float = 0.60,
        **kwargs
    ):
        super().__init__()
        # If folds passed as int or list:
        if folds is not None:
            if isinstance(folds, int):
                folds_total = folds
            elif isinstance(folds, list):
                if not fold_details:
                    fold_details = folds
                folds_total = len(folds)
        if fold_details is None:
            fold_details = []
        if folds_total == 0 and fold_details:
            folds_total = len(fold_details)
        if folds_passed == 0 and fold_details:
            folds_passed = sum(1 for f in fold_details if f.get("passed", False) or f.get("wfe", 0.0) >= wfe_threshold)

        wfe_val = round(float(wfe), 4)
        pass_val = bool(passed if passed is not None else (wfe_val >= wfe_threshold))
        robust_val = bool(is_robust if is_robust is not None else pass_val)

        data = {
            "wfe": wfe_val,
            "walk_forward_efficiency": wfe_val,
            "wfe_pct": round(wfe_val * 100.0, 2),
            "passed": pass_val,
            "is_robust": robust_val,
            "wfe_threshold": float(wfe_threshold),
            "folds_total": int(folds_total),
            "folds_passed": int(folds_passed),
            "folds": fold_details if fold_details else folds_total,
            "fold_details": fold_details,
            "splits_results": fold_details,
            "mean_is_return": round(float(mean_is_return), 4),
            "in_sample_return": round(float(mean_is_return), 4),
            "mean_oos_return": round(float(mean_oos_return), 4),
            "out_of_sample_return": round(float(mean_oos_return), 4),
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "monte_carlo": monte_carlo,
            "parameter_stability": round(float(parameter_stability), 4),
            "summary": f"Walk-Forward Validation: Folds={folds_total}, WFE={wfe_val:.2%}, Passed={pass_val}",
        }
        data.update(kwargs)
        for k, v in data.items():
            self[k] = v

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(f"'WalkForwardResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class WalkForwardValidator:
    """
    Anti-Overfitting Validation Engine for GBPJPY Quantitative Trading Framework.
    Implements 4-tier anti-overfitting protocol adhering to institutional standards:
      1. Strict chronological IS/OOS partitioning (zero data leakage)
      2. Rolling walk-forward window generation (24 folds)
      3. Walk-Forward Efficiency (WFE) ratio calculation (>=60% threshold)
      4. Monte Carlo trade sequence permutation & drawdown modeling
      5. Parameter stability plateau scoring
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        in_sample_bars: int = 1000,
        out_of_sample_bars: int = 250,
        n_splits: Optional[int] = None,
        folds: Optional[int] = None,
        is_ratio: float = 0.70,
        wfe_threshold: float = 0.60,
        initial_balance: float = 250.0,
        strategy: Optional[Any] = None,
        engine: Optional[Any] = None,
        monte_carlo_iterations: int = 500,
        **kwargs
    ):
        self.config = config
        self.in_sample_bars = int(in_sample_bars)
        self.out_of_sample_bars = int(out_of_sample_bars)
        self.folds = self.n_splits = int(folds if folds is not None else (n_splits if n_splits is not None else 10))
        self.is_ratio = float(is_ratio)
        self.wfe_threshold = float(wfe_threshold)
        self.initial_balance = float(initial_balance)
        self.strategy = strategy
        self.engine = engine
        self.monte_carlo_iterations = int(monte_carlo_iterations)

        if self.config is not None:
            risk_cfg = getattr(self.config, "risk_management", getattr(self.config, "risk", None))
            if risk_cfg is not None:
                self.initial_balance = float(getattr(risk_cfg, "initial_balance", self.initial_balance))

    def split_is_oos(
        self,
        df: pd.DataFrame,
        is_ratio: float = 0.70,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strict chronological partition into In-Sample and Out-of-Sample datasets.
        Guarantees zero data leakage (max timestamp in IS < min timestamp in OOS).
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(), pd.DataFrame()

        df_copy = df.copy()
        if "time" in df_copy.columns:
            if df_copy.index.name == "time":
                df_copy.index.name = None
            df_sorted = df_copy.sort_values(by="time").reset_index(drop=True)
        else:
            df_sorted = df_copy.sort_index().reset_index(drop=True)

        total_rows = len(df_sorted)
        ratio = max(0.01, min(0.99, float(is_ratio)))
        split_idx = int(round(total_rows * ratio))
        split_idx = max(1, min(total_rows - 1, split_idx))

        df_is = df_sorted.iloc[:split_idx].copy().reset_index(drop=True)
        df_oos = df_sorted.iloc[split_idx:].copy().reset_index(drop=True)
        return df_is, df_oos

    train_test_split = split_is_oos  # Alias

    def generate_windows(
        self,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str],
        folds: int = 24,
        rolling: bool = True,
    ) -> List[WalkForwardWindow]:
        """
        Generates exactly `folds` rolling or expanding walk-forward windows across [start_date, end_date].
        Each window strictly satisfies: train_start < train_end <= test_start < test_end.
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).to_pydatetime()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).to_pydatetime()

        if start_date >= end_date:
            raise ValueError(f"start_date ({start_date}) must precede end_date ({end_date})")

        k_folds = max(1, int(folds))
        total_duration = end_date - start_date

        k_train = max(1, k_folds // 3)
        total_units = k_folds + k_train
        unit_delta = total_duration / total_units

        windows: List[WalkForwardWindow] = []
        for i in range(k_folds):
            if rolling:
                train_start = start_date + i * unit_delta
                train_end = train_start + k_train * unit_delta
            else:
                train_start = start_date
                train_end = start_date + (k_train + i) * unit_delta

            test_start = train_end
            test_end = test_start + unit_delta
            if i == k_folds - 1:
                test_end = end_date  # Exact terminal clamp

            windows.append(WalkForwardWindow(
                fold=i + 1,
                fold_idx=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
        return windows

    generate_walk_forward_windows = generate_windows  # Alias

    def generate_splits(
        self,
        df: pd.DataFrame,
        in_sample_bars: Optional[int] = None,
        out_of_sample_bars: Optional[int] = None,
        n_splits: Optional[int] = None,
        folds: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates rolling chronological (IS, OOS) DataFrame splits with zero lookahead overlap.
        Guarantees: oos_min_time >= is_max_time.
        """
        if df is None or len(df) == 0:
            return []

        df_copy = df.copy()
        if "time" in df_copy.columns:
            if df_copy.index.name == "time":
                df_copy.index.name = None
            df_sorted = df_copy.sort_values(by="time").reset_index(drop=True)
        else:
            df_sorted = df_copy.sort_index().reset_index(drop=True)

        total_bars = len(df_sorted)
        is_bars = int(in_sample_bars if in_sample_bars is not None else self.in_sample_bars)
        oos_bars = int(out_of_sample_bars if out_of_sample_bars is not None else self.out_of_sample_bars)
        target_splits = int(folds if folds is not None else (n_splits if n_splits is not None else self.folds))

        if total_bars < is_bars + oos_bars:
            df_is, df_oos = self.split_is_oos(df_sorted, is_ratio=self.is_ratio)
            return [(df_is, df_oos)]

        window_size = is_bars + oos_bars
        available = total_bars - window_size
        step = max(1, available // max(1, target_splits - 1)) if target_splits > 1 else available

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for k in range(target_splits):
            start_idx = k * step
            is_end = start_idx + is_bars
            oos_end = is_end + oos_bars
            if oos_end > total_bars:
                break
            is_data = df_sorted.iloc[start_idx:is_end].copy().reset_index(drop=True)
            oos_data = df_sorted.iloc[is_end:oos_end].copy().reset_index(drop=True)
            splits.append((is_data, oos_data))

        return splits

    create_windows = generate_splits  # Alias

    def calculate_wfe(
        self,
        is_return: Optional[float] = None,
        oos_return: Optional[float] = None,
        is_annual_return: Optional[float] = None,
        oos_annual_return: Optional[float] = None,
        as_percentage: bool = False,
        **kwargs
    ) -> float:
        """
        Calculates Walk-Forward Efficiency (WFE) ratio:
        WFE = Annualized OOS Return / Annualized IS Return.
        Accepts both is_return/oos_return and is_annual_return/oos_annual_return.
        Clamps to 0.0 if IS return <= 1e-9 or OOS return <= 0.0.
        """
        is_ret = is_annual_return if is_annual_return is not None else is_return
        oos_ret = oos_annual_return if oos_annual_return is not None else oos_return

        if is_ret is None or oos_ret is None:
            return 0.0

        is_ret = float(is_ret)
        oos_ret = float(oos_ret)

        if is_ret <= 1e-9 or oos_ret <= 0.0:
            wfe = 0.0
        else:
            wfe = oos_ret / is_ret

        return round(wfe * 100.0, 4) if as_percentage else round(wfe, 4)

    calculate_walk_forward_efficiency = calculate_wfe  # Alias

    def monte_carlo_simulation(
        self,
        trades_pnl: Sequence[Union[float, Dict[str, Any]]],
        initial_balance: Optional[float] = None,
        iterations: int = 500,
        seed: Optional[int] = 42,
        **kwargs
    ) -> MonteCarloResult:
        """
        Runs Monte Carlo trade permutation test by randomly shuffling trade execution order.
        Strictly preserves terminal return mean while estimating drawdown distribution.
        """
        init_bal = float(initial_balance if initial_balance is not None else self.initial_balance)
        num_iter = max(1, int(iterations))

        clean_pnls: List[float] = []
        for t in trades_pnl:
            if isinstance(t, dict):
                clean_pnls.append(float(t.get("net_pnl_usd", t.get("pnl", 0.0))))
            else:
                clean_pnls.append(float(t))

        if not clean_pnls:
            return MonteCarloResult(
                mean_terminal_equity=init_bal,
                worst_drawdown=0.0,
                median_drawdown=0.0,
                p95_drawdown=0.0,
                p99_drawdown=0.0,
                ruin_probability=0.0,
                iterations=num_iter,
                initial_balance=init_bal,
            )

        rng = np.random.RandomState(seed if seed is not None else 42)
        pnl_arr = np.array(clean_pnls, dtype=np.float64)
        total_pnl = float(np.sum(pnl_arr))
        mean_terminal = round(init_bal + total_pnl, 4)

        max_dds_dollars: List[float] = []
        max_dds_pct: List[float] = []
        terminal_equities: List[float] = []
        ruin_count = 0

        for _ in range(num_iter):
            perm = rng.permutation(pnl_arr)
            curve = init_bal + np.concatenate(([0.0], np.cumsum(perm)))
            terminal_equities.append(float(curve[-1]))

            peaks = np.maximum.accumulate(curve)
            dd_dollars = peaks - curve
            dd_pct = np.where(peaks > 0, dd_dollars / peaks * 100.0, 0.0)

            run_max_dd_dollars = float(np.max(dd_dollars))
            run_max_dd_pct = float(np.max(dd_pct))

            max_dds_dollars.append(run_max_dd_dollars)
            max_dds_pct.append(run_max_dd_pct)

            if float(np.min(curve)) <= 0.0:
                ruin_count += 1

        worst_dd = float(np.max(max_dds_dollars))
        median_dd = float(np.median(max_dds_dollars))
        p95_dd = float(np.percentile(max_dds_dollars, 95.0))
        p99_dd = float(np.percentile(max_dds_dollars, 99.0))
        ruin_prob = float(ruin_count / num_iter)

        return MonteCarloResult(
            mean_terminal_equity=mean_terminal,
            worst_drawdown=round(worst_dd, 4),
            median_drawdown=round(median_dd, 4),
            p95_drawdown=round(p95_dd, 4),
            p99_drawdown=round(p99_dd, 4),
            ruin_probability=round(ruin_prob, 4),
            iterations=num_iter,
            initial_balance=init_bal,
            terminal_equities=terminal_equities,
            drawdowns=max_dds_dollars,
        )

    permute_trades = monte_carlo_simulation  # Alias

    def calculate_parameter_stability(
        self,
        param_grid_results: Dict[Any, float],
        **kwargs
    ) -> float:
        """
        Evaluates parameter robustness across neighboring grid configurations.
        Returns a score in (0.0, 1.0] where higher values indicate an optimal plateau
        rather than an overfitted parameter spike.
        """
        if not param_grid_results:
            return 0.0

        best_param, best_val = max(param_grid_results.items(), key=lambda x: x[1])
        if best_val <= 0:
            return 0.0

        dim = len(best_param) if isinstance(best_param, tuple) else 1
        if isinstance(best_param, tuple):
            unique_coords = [sorted(list(set(k[d] for k in param_grid_results.keys()))) for d in range(dim)]
            best_indices = [unique_coords[d].index(best_param[d]) for d in range(dim)]
            neighbor_vals = []
            for k, v in param_grid_results.items():
                if k == best_param:
                    continue
                k_indices = [unique_coords[d].index(k[d]) for d in range(dim)]
                if all(abs(k_indices[d] - best_indices[d]) <= 1 for d in range(dim)):
                    neighbor_vals.append(v)
        else:
            neighbor_vals = [v for k, v in param_grid_results.items() if k != best_param]

        if not neighbor_vals:
            neighbor_vals = [v for k, v in param_grid_results.items() if k != best_param]

        if not neighbor_vals:
            return 1.0

        mean_neighbor = float(np.mean(neighbor_vals))
        score = max(0.01, min(1.0, mean_neighbor / best_val))
        return round(float(score), 4)

    evaluate_parameter_stability = calculate_parameter_stability  # Alias
    parameter_stability = calculate_parameter_stability           # Alias

    def run(
        self,
        df: pd.DataFrame,
        strategy: Optional[Any] = None,
        symbol: str = "GBPJPY",
        **kwargs
    ) -> WalkForwardResult:
        """
        Executes complete rolling Walk-Forward analysis across generated splits.
        """
        if df is None or len(df) == 0:
            return WalkForwardResult(
                wfe=0.0, passed=False, summary="Empty DataFrame supplied."
            )

        # Ensure indicators are computed
        if DataEngine is not None and "atr" not in df.columns:
            de = DataEngine()
            df = de.compute_indicators(df)

        splits = self.generate_splits(df)
        if not splits:
            splits = [self.split_is_oos(df, is_ratio=self.is_ratio)]

        strat = strategy or self.strategy
        if strat is None and GBPJPYDualRegimeScalper is not None:
            strat = GBPJPYDualRegimeScalper(config=self.config)

        splits_results: List[Dict[str, Any]] = []
        is_returns: List[float] = []
        oos_returns: List[float] = []
        all_oos_trades: List[Any] = []

        for fold_idx, (is_df, oos_df) in enumerate(splits):
            if BacktestEngine is not None and strat is not None:
                engine_is = BacktestEngine(
                    config=self.config,
                    initial_balance=self.initial_balance,
                    symbol=symbol,
                    strategy=strat,
                )
                is_res = engine_is.run(is_df, strategy=strat)

                engine_oos = BacktestEngine(
                    config=self.config,
                    initial_balance=self.initial_balance,
                    symbol=symbol,
                    strategy=strat,
                )
                oos_res = engine_oos.run(oos_df, strategy=strat)

                is_ret = (is_res.final_balance - is_res.initial_balance) / is_res.initial_balance
                oos_ret = (oos_res.final_balance - oos_res.initial_balance) / oos_res.initial_balance
                trades_oos = oos_res.trades
            else:
                is_ret = 0.0
                oos_ret = 0.0
                trades_oos = []

            fold_wfe = self.calculate_wfe(is_return=is_ret, oos_return=oos_ret)
            is_returns.append(is_ret)
            oos_returns.append(oos_ret)
            all_oos_trades.extend(trades_oos)

            splits_results.append({
                "fold": fold_idx + 1,
                "is_bars": len(is_df),
                "oos_bars": len(oos_df),
                "is_return": round(is_ret, 4),
                "oos_return": round(oos_ret, 4),
                "wfe": fold_wfe,
                "passed": bool(fold_wfe >= self.wfe_threshold),
            })

        mean_is = float(np.mean(is_returns)) if is_returns else 0.0
        mean_oos = float(np.mean(oos_returns)) if oos_returns else 0.0

        overall_wfe = self.calculate_wfe(is_return=mean_is, oos_return=mean_oos)
        passed_folds = sum(1 for f in splits_results if f["passed"])
        passed = bool(overall_wfe >= self.wfe_threshold)

        mc_pnls = [t.get("net_pnl_usd", 0.0) if isinstance(t, dict) else float(t) for t in all_oos_trades]
        mc_result = self.monte_carlo_simulation(mc_pnls, initial_balance=self.initial_balance)

        return WalkForwardResult(
            wfe=overall_wfe,
            passed=passed,
            folds_total=len(splits),
            folds_passed=passed_folds,
            mean_is_return=mean_is,
            mean_oos_return=mean_oos,
            fold_details=splits_results,
            monte_carlo=mc_result,
            is_robust=passed,
            wfe_threshold=self.wfe_threshold,
        )
