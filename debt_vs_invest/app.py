#!/usr/bin/env python3
"""
debt_vs_invest.py

Compare two strategies over a horizon:
  A) Use extra monthly cash to pay down debt faster (then invest freed cash)
  B) Invest extra monthly cash while paying debt on schedule (then invest freed cash)

Supports:
- Amortizing debt (fixed rate, fixed remaining term)
- Taxable / Traditional / Roth-ish investment tax treatments (simplified)
- Interest deductibility (simplified; user supplies effective deductible fraction)
- Inflation adjustments (real vs nominal)
- Monte Carlo market paths + summary statistics
- Streamlit GUI with tooltips + interactive hover + median bands + difference chart:
    streamlit run debt_vs_invest.py

Disclaimer: This is a simplified educational model. Not tax/legal/financial advice.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt

# Dark plot styling for CLI plots
plt.style.use("dark_background")


# -----------------------------
# Utilities
# -----------------------------


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def monthly_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """Standard fixed-rate amortizing payment."""
    if term_months <= 0:
        return 0.0
    if abs(annual_rate) < 1e-12:
        return principal / term_months
    r = annual_rate / 12.0
    return principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)


def ann_arith_to_lognormal_params(
    mu_ann: float, sigma_ann: float
) -> Tuple[float, float]:
    """
    Convert annual arithmetic mean return (mu_ann) and annual stdev (sigma_ann)
    into annual lognormal parameters for gross return G = 1 + R.

    If ln(G) ~ N(a, b^2), then:
      E[G] = exp(a + b^2/2) = 1 + mu_ann
      SD[G] = sigma_ann  (approx; treats R stdev as G stdev)
    """
    mu_g = 1.0 + mu_ann
    sd_g = sigma_ann
    if sd_g < 1e-12:
        return math.log(max(mu_g, 1e-12)), 0.0

    var_g = sd_g**2
    b2 = math.log(1.0 + var_g / (mu_g**2))
    a = math.log(mu_g) - 0.5 * b2
    return a, math.sqrt(b2)


def annual_to_monthly_lognormal(mu_ann: float, sigma_ann: float) -> Tuple[float, float]:
    """
    Assuming iid monthly log-returns with constant parameters:
      ln(G_year) = sum ln(G_month)
    So:
      a_month = a_year / 12
      b_month = b_year / sqrt(12)
    """
    a_y, b_y = ann_arith_to_lognormal_params(mu_ann, sigma_ann)
    return a_y / 12.0, b_y / math.sqrt(12.0)


# -----------------------------
# Parameters
# -----------------------------


@dataclass
class DebtParams:
    balance: float = 400_000.0
    annual_rate: float = 0.065
    remaining_term_months: int = 360
    min_payment_override: Optional[float] = None


@dataclass
class TaxParams:
    # Interest deduction (simplified)
    interest_deductible_fraction: float = 0.0
    marginal_income_tax_rate: float = 0.24

    # Investment taxes (simplified)
    account_type: str = "taxable"  # taxable | traditional | roth
    dividend_tax_rate: float = 0.15
    capital_gains_tax_rate: float = 0.15
    ordinary_withdrawal_tax_rate: float = 0.24  # for traditional at withdrawal


@dataclass
class InvestParams:
    exp_return_annual: float = 0.07
    vol_annual: float = 0.15
    dividend_yield_annual: float = 0.015
    stochastic: bool = True


@dataclass
class ScenarioParams:
    horizon_years: float = 30.0
    inflation_annual: float = 0.025
    monthly_surplus: float = 1_000.0
    n_sims: int = 10_000
    seed: int = 42
    risk_aversion_gamma: float = 0.0


# -----------------------------
# Core simulation
# -----------------------------


class ReplayRNG:
    """
    Minimal RNG wrapper so we can replay the same standard_normal draws for both strategies.
    Only implements standard_normal(size=None).
    """

    def __init__(self, shocks: Optional[np.ndarray]):
        self.shocks = shocks
        self.i = 0

    def standard_normal(self, size=None):
        if self.shocks is None:
            raise RuntimeError(
                "ReplayRNG used but shocks are None (deterministic mode)."
            )
        if size is None:
            if self.i >= len(self.shocks):
                raise IndexError("No more shocks.")
            x = self.shocks[self.i]
            self.i += 1
            return x
        n = int(np.prod(size))
        if self.i + n > len(self.shocks):
            raise IndexError("Not enough shocks.")
        xs = self.shocks[self.i : self.i + n]
        self.i += n
        return xs.reshape(size)


def simulate_one_path(
    debt: DebtParams,
    invest: InvestParams,
    tax: TaxParams,
    scen: ScenarioParams,
    strategy: str,
    rng: ReplayRNG | np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    strategy:
      - "pay_down": allocate as much as possible to debt (min + surplus), then invest once debt is gone
      - "invest": pay only minimum to debt, invest the surplus; once debt is gone (scheduled), invest freed min payment too
    """
    assert strategy in ("pay_down", "invest")

    months = max(int(round(scen.horizon_years * 12)), 1)

    bal = max(float(debt.balance), 0.0)
    r_debt_m = debt.annual_rate / 12.0

    if debt.min_payment_override is not None:
        min_pmt = float(debt.min_payment_override)
    else:
        min_pmt = monthly_payment(bal, debt.annual_rate, debt.remaining_term_months)

    monthly_available = min_pmt + scen.monthly_surplus
    value = 0.0

    if invest.stochastic:
        a_m, b_m = annual_to_monthly_lognormal(
            invest.exp_return_annual, invest.vol_annual
        )
    else:
        det_gross_m = (1.0 + invest.exp_return_annual) ** (1.0 / 12.0)

    div_y_m = invest.dividend_yield_annual / 12.0
    infl_gross_m = (1.0 + scen.inflation_annual) ** (1.0 / 12.0)

    debt_balance = np.zeros(months + 1)
    inv_value = np.zeros(months + 1)
    net_worth = np.zeros(months + 1)
    net_worth_real = np.zeros(months + 1)
    infl_index = np.ones(months + 1)

    debt_balance[0] = bal
    inv_value[0] = value
    net_worth[0] = value - bal
    net_worth_real[0] = net_worth[0]

    remaining_term = int(debt.remaining_term_months)

    for t in range(1, months + 1):
        infl_index[t] = infl_index[t - 1] * infl_gross_m

        debt_exists = bal > 1e-8 and remaining_term > 0
        req_min = min_pmt if debt_exists else 0.0

        interest = bal * r_debt_m if debt_exists else 0.0

        # Interest deduction benefit (simplified)
        ded_frac = clamp(tax.interest_deductible_fraction, 0.0, 1.0)
        mtr = clamp(tax.marginal_income_tax_rate, 0.0, 0.60)
        tax_savings = interest * ded_frac * mtr

        # Debt payment
        if debt_exists:
            if strategy == "pay_down":
                pay_budget = monthly_available
            else:
                pay_budget = req_min

            pay_budget = max(pay_budget, req_min)
            payment = pay_budget

            principal_pay = payment - interest
            bal = bal - principal_pay

            leftover = 0.0
            if bal <= 0.0:
                leftover = -bal
                bal = 0.0
                remaining_term = 0
            else:
                remaining_term -= 1
        else:
            payment = 0.0
            leftover = 0.0

        invest_contrib = (monthly_available - payment) + leftover + tax_savings
        invest_contrib = max(invest_contrib, 0.0)

        # Market return
        if invest.stochastic:
            gross = float(np.exp(a_m + b_m * float(rng.standard_normal())))
        else:
            gross = float(det_gross_m)

        # Split into dividend + price return (simple approximation)
        div = div_y_m
        price_return = (gross - 1.0) - div

        acct = tax.account_type.lower().strip()

        if acct == "taxable":
            # No running liquidation taxes here; terminal metrics apply end-of-horizon taxes separately.
            value = value * (1.0 + price_return)
            div_cash = value * div
            div_tax = div_cash * clamp(tax.dividend_tax_rate, 0.0, 0.60)
            div_net = div_cash - div_tax
            value += div_net
            value += invest_contrib

        elif acct == "traditional":
            value = value * gross + invest_contrib

        elif acct == "roth":
            value = value * gross + invest_contrib

        else:
            raise ValueError(
                f"Unknown account_type: {tax.account_type!r} (use taxable|traditional|roth)"
            )

        debt_balance[t] = bal
        inv_value[t] = value

        nw = value - bal
        net_worth[t] = nw
        net_worth_real[t] = nw / infl_index[t]

    return {
        "debt_balance": debt_balance,
        "inv_value": inv_value,
        "net_worth": net_worth,
        "net_worth_real": net_worth_real,
        "infl_index": infl_index,
        "min_payment": np.array([min_pmt], dtype=float),
    }


def liquidate_after_tax(value: float, basis: float, tax: TaxParams) -> float:
    """Apply end-of-horizon taxes on the investment account (simplified)."""
    acct = tax.account_type.lower().strip()

    if acct == "taxable":
        gains = max(value - basis, 0.0)
        cg_tax = gains * clamp(tax.capital_gains_tax_rate, 0.0, 0.60)
        return value - cg_tax

    if acct == "traditional":
        return value * (1.0 - clamp(tax.ordinary_withdrawal_tax_rate, 0.0, 0.60))

    if acct == "roth":
        return value

    raise ValueError(
        f"Unknown account_type: {tax.account_type!r} (use taxable|traditional|roth)"
    )


def replay_taxable_terminal_value_and_basis(
    debt: DebtParams,
    invest: InvestParams,
    tax: TaxParams,
    scen: ScenarioParams,
    strategy: str,
    shocks: Optional[np.ndarray],
) -> Tuple[float, float]:
    """Replays one path and returns (terminal_value, terminal_basis) for taxable accounts."""
    months = max(int(round(scen.horizon_years * 12)), 1)

    bal = float(debt.balance)
    r_debt_m = debt.annual_rate / 12.0

    if debt.min_payment_override is not None:
        min_pmt = float(debt.min_payment_override)
    else:
        min_pmt = monthly_payment(bal, debt.annual_rate, debt.remaining_term_months)

    monthly_available = min_pmt + scen.monthly_surplus

    value = 0.0
    basis = 0.0

    if invest.stochastic:
        a_m, b_m = annual_to_monthly_lognormal(
            invest.exp_return_annual, invest.vol_annual
        )
        assert shocks is not None and len(shocks) >= months
    else:
        det_gross_m = (1.0 + invest.exp_return_annual) ** (1.0 / 12.0)

    div_y_m = invest.dividend_yield_annual / 12.0
    remaining_term = int(debt.remaining_term_months)

    for t in range(months):
        debt_exists = bal > 1e-8 and remaining_term > 0
        req_min = min_pmt if debt_exists else 0.0
        interest = bal * r_debt_m if debt_exists else 0.0

        ded_frac = clamp(tax.interest_deductible_fraction, 0.0, 1.0)
        mtr = clamp(tax.marginal_income_tax_rate, 0.0, 0.60)
        tax_savings = interest * ded_frac * mtr

        if debt_exists:
            if strategy == "pay_down":
                pay_budget = monthly_available
            else:
                pay_budget = req_min
            pay_budget = max(pay_budget, req_min)

            payment = pay_budget
            principal_pay = payment - interest
            bal = bal - principal_pay

            leftover = 0.0
            if bal <= 0.0:
                leftover = -bal
                bal = 0.0
                remaining_term = 0
            else:
                remaining_term -= 1
        else:
            payment = 0.0
            leftover = 0.0

        invest_contrib = (monthly_available - payment) + leftover + tax_savings
        invest_contrib = max(invest_contrib, 0.0)

        if invest.stochastic:
            z = shocks[t]
            gross = float(np.exp(a_m + b_m * z))
        else:
            gross = float(det_gross_m)

        div = div_y_m
        price_return = (gross - 1.0) - div

        value = value * (1.0 + price_return)

        div_cash = value * div
        div_tax = div_cash * clamp(tax.dividend_tax_rate, 0.0, 0.60)
        div_net = div_cash - div_tax
        value += div_net
        basis += div_net

        value += invest_contrib
        basis += invest_contrib

    return value, basis


# -----------------------------
# Stats / Reporting helpers
# -----------------------------


def summarize(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
    }


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x/1e6:,.2f}M"
    if x >= 1e3:
        return f"{sign}${x:,.0f}"
    return f"{sign}${x:,.2f}"


def certainty_equivalent(wealth: np.ndarray, gamma: float) -> float:
    """
    CRRA certainty equivalent on *real* terminal wealth.
    Clamps wealth to >= 1 to avoid log/negative issues.
    """
    w = np.asarray(wealth, dtype=float)
    w = np.maximum(w, 1.0)

    if abs(gamma - 1.0) < 1e-9:
        return float(np.exp(np.mean(np.log(w))))

    u = (w ** (1.0 - gamma)) / (1.0 - gamma)
    eu = float(np.mean(u))
    return float(((1.0 - gamma) * eu) ** (1.0 / (1.0 - gamma)))


# -----------------------------
# Main simulation
# -----------------------------


def run_simulation(
    debt: DebtParams,
    invest: InvestParams,
    tax: TaxParams,
    scen: ScenarioParams,
    *,
    path_sample_size: int = 2000,  # for median/band charts (reservoir sample)
) -> Dict[str, object]:
    """
    Returns terminal distributions + a representative path + a reservoir sample of paths
    for median + percentile bands over time.
    """
    rng = np.random.default_rng(scen.seed)
    rng_res = np.random.default_rng(
        scen.seed + 12345
    )  # independent RNG for reservoir sampling

    months = max(int(round(scen.horizon_years * 12)), 1)
    n = int(max(scen.n_sims, 1))

    terminal_paydown = np.zeros(n, dtype=float)
    terminal_invest = np.zeros(n, dtype=float)
    terminal_paydown_real = np.zeros(n, dtype=float)
    terminal_invest_real = np.zeros(n, dtype=float)

    rep_paths: Dict[str, Dict[str, np.ndarray]] = {}

    # Reservoir sample of real net worth paths (for bands)
    k = int(min(max(path_sample_size, 0), n))
    sample_pay_real = np.zeros((k, months + 1), dtype=np.float32) if k > 0 else None
    sample_inv_real = np.zeros((k, months + 1), dtype=np.float32) if k > 0 else None

    for i in range(n):
        shocks = rng.standard_normal(months) if invest.stochastic else None

        out_a = simulate_one_path(
            debt,
            invest,
            tax,
            scen,
            "pay_down",
            ReplayRNG(shocks) if shocks is not None else rng,
        )
        out_b = simulate_one_path(
            debt,
            invest,
            tax,
            scen,
            "invest",
            ReplayRNG(shocks) if shocks is not None else rng,
        )

        # Terminal after-tax liquidation
        if tax.account_type.lower().strip() == "taxable":
            v_a, b_a = replay_taxable_terminal_value_and_basis(
                debt, invest, tax, scen, "pay_down", shocks
            )
            v_b, b_b = replay_taxable_terminal_value_and_basis(
                debt, invest, tax, scen, "invest", shocks
            )
            after_tax_a = liquidate_after_tax(v_a, b_a, tax)
            after_tax_b = liquidate_after_tax(v_b, b_b, tax)
        else:
            after_tax_a = liquidate_after_tax(float(out_a["inv_value"][-1]), 0.0, tax)
            after_tax_b = liquidate_after_tax(float(out_b["inv_value"][-1]), 0.0, tax)

        nw_a = after_tax_a - float(out_a["debt_balance"][-1])
        nw_b = after_tax_b - float(out_b["debt_balance"][-1])

        terminal_paydown[i] = nw_a
        terminal_invest[i] = nw_b
        terminal_paydown_real[i] = nw_a / float(out_a["infl_index"][-1])
        terminal_invest_real[i] = nw_b / float(out_b["infl_index"][-1])

        if i == 0:
            rep_paths = {"pay_down": out_a, "invest": out_b}

        # Reservoir sample for time-series bands (unbiased sample of sims)
        if k > 0:
            if i < k:
                idx = i
            else:
                j = int(rng_res.integers(0, i + 1))
                idx = j if j < k else None
            if idx is not None:
                sample_pay_real[idx, :] = out_a["net_worth_real"].astype(
                    np.float32, copy=False
                )
                sample_inv_real[idx, :] = out_b["net_worth_real"].astype(
                    np.float32, copy=False
                )

    diff = terminal_invest - terminal_paydown
    diff_real = terminal_invest_real - terminal_paydown_real

    results: Dict[str, object] = {
        "terminal_paydown": terminal_paydown,
        "terminal_invest": terminal_invest,
        "terminal_paydown_real": terminal_paydown_real,
        "terminal_invest_real": terminal_invest_real,
        "diff": diff,
        "diff_real": diff_real,
        "rep_paths": rep_paths,
        "min_payment": (
            float(rep_paths["pay_down"]["min_payment"][0]) if rep_paths else None
        ),
        # Samples for bands
        "path_sample_paydown_real": sample_pay_real,
        "path_sample_invest_real": sample_inv_real,
        "path_sample_size": k,
    }

    gamma = scen.risk_aversion_gamma
    if gamma and gamma > 0:
        results["ce_paydown_real"] = certainty_equivalent(terminal_paydown_real, gamma)
        results["ce_invest_real"] = certainty_equivalent(terminal_invest_real, gamma)
        results["ce_diff_real"] = float(results["ce_invest_real"]) - float(
            results["ce_paydown_real"]
        )

    return results


# -----------------------------
# CLI
# -----------------------------


def print_report(results: Dict[str, object], scen: ScenarioParams):
    term_a = np.asarray(results["terminal_paydown"])
    term_b = np.asarray(results["terminal_invest"])
    diff = np.asarray(results["diff"])
    diff_r = np.asarray(results["diff_real"])

    sa = summarize(term_a)
    sb = summarize(term_b)
    sd = summarize(diff)
    sdr = summarize(diff_r)

    p_win = float(np.mean(diff > 0.0))
    p_win_real = float(np.mean(diff_r > 0.0))

    print("\n=== Debt vs Invest Simulation ===")
    print(
        f"Horizon: {scen.horizon_years:.2f} years | Sims: {scen.n_sims:,} | Inflation: {scen.inflation_annual*100:.2f}%/yr"
    )
    print(f"Monthly surplus: {money(scen.monthly_surplus)}")
    print("A: Pay down debt faster, then invest freed cash")
    print("B: Invest surplus while paying debt on schedule, then invest freed cash\n")

    print("--- Terminal Net Worth (Nominal, after end-of-horizon account taxes) ---")
    print(
        f"A Pay-down : mean {money(sa['mean'])} | median {money(sa['median'])} | p10 {money(sa['p10'])} | p90 {money(sa['p90'])}"
    )
    print(
        f"B Invest   : mean {money(sb['mean'])} | median {money(sb['median'])} | p10 {money(sb['p10'])} | p90 {money(sb['p90'])}"
    )
    print(
        f"Diff (B-A) : mean {money(sd['mean'])} | median {money(sd['median'])} | p10 {money(sd['p10'])} | p90 {money(sd['p90'])}"
    )
    print(f"P(B beats A): {p_win*100:.1f}%\n")

    print("--- Terminal Net Worth (Real, inflation-adjusted) ---")
    print(
        f"Diff_real (B-A): mean {money(sdr['mean'])} | median {money(sdr['median'])} | p10 {money(sdr['p10'])} | p90 {money(sdr['p90'])}"
    )
    print(f"P(B beats A) in real terms: {p_win_real*100:.1f}%\n")

    if "ce_diff_real" in results:
        print("--- Risk-adjusted (CRRA certainty equivalent, real terminal wealth) ---")
        print(f"Risk aversion gamma: {scen.risk_aversion_gamma}")
        print(f"CE(A) real: {money(float(results['ce_paydown_real']))}")
        print(f"CE(B) real: {money(float(results['ce_invest_real']))}")
        print(f"CE diff (B-A) real: {money(float(results['ce_diff_real']))}\n")


def plot_rep_paths(
    results: Dict[str, object],
    scen: ScenarioParams,
    show: bool = True,
    save_path: Optional[str] = None,
):
    rep = results.get("rep_paths", {})
    if not rep:
        return

    pay = rep["pay_down"]
    inv = rep["invest"]

    months = len(pay["net_worth"]) - 1
    t = np.arange(months + 1) / 12.0

    plt.figure()
    plt.plot(t, pay["debt_balance"], label="Debt balance (pay-down)")
    plt.plot(t, inv["debt_balance"], label="Debt balance (invest)")
    plt.xlabel("Years")
    plt.ylabel("Debt balance ($)")
    plt.legend()
    plt.title("Debt Balance Over Time (Representative Path)")

    plt.figure()
    plt.plot(t, pay["net_worth_real"], label="Real net worth (pay-down)")
    plt.plot(t, inv["net_worth_real"], label="Real net worth (invest)")
    plt.xlabel("Years")
    plt.ylabel("Real net worth ($)")
    plt.legend()
    plt.title("Real Net Worth Over Time (Representative Path)")

    plt.figure()
    diff_real = np.asarray(results["diff_real"])
    plt.hist(diff_real, bins=60)
    plt.xlabel("Terminal real net worth difference (B - A)")
    plt.ylabel("Count")
    plt.title("Distribution of Terminal Outcomes (Real, B - A)")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        for i, fig_num in enumerate(plt.get_fignums(), start=1):
            plt.figure(fig_num)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"plot_{i}.png"), dpi=160)

    if show:
        plt.show()
    else:
        plt.close("all")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare paying down debt vs investing surplus cash."
    )
    # Debt
    p.add_argument("--debt-balance", type=float, default=DebtParams.balance)
    p.add_argument(
        "--debt-rate",
        type=float,
        default=DebtParams.annual_rate,
        help="Annual interest rate (e.g. 0.065)",
    )
    p.add_argument(
        "--debt-term-years",
        type=float,
        default=DebtParams.remaining_term_months / 12.0,
        help="Remaining term in years",
    )
    p.add_argument(
        "--min-payment",
        type=float,
        default=None,
        help="Override minimum monthly payment",
    )
    # Scenario
    p.add_argument("--horizon-years", type=float, default=ScenarioParams.horizon_years)
    p.add_argument(
        "--monthly-surplus", type=float, default=ScenarioParams.monthly_surplus
    )
    p.add_argument("--inflation", type=float, default=ScenarioParams.inflation_annual)
    # Market
    p.add_argument("--exp-return", type=float, default=InvestParams.exp_return_annual)
    p.add_argument("--vol", type=float, default=InvestParams.vol_annual)
    p.add_argument(
        "--div-yield", type=float, default=InvestParams.dividend_yield_annual
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic returns instead of Monte Carlo",
    )
    # Taxes
    p.add_argument(
        "--account-type",
        type=str,
        default=TaxParams.account_type,
        choices=["taxable", "traditional", "roth"],
    )
    p.add_argument(
        "--interest-deductible-fraction",
        type=float,
        default=TaxParams.interest_deductible_fraction,
    )
    p.add_argument(
        "--marginal-tax", type=float, default=TaxParams.marginal_income_tax_rate
    )
    p.add_argument("--div-tax", type=float, default=TaxParams.dividend_tax_rate)
    p.add_argument("--cg-tax", type=float, default=TaxParams.capital_gains_tax_rate)
    p.add_argument(
        "--withdraw-tax", type=float, default=TaxParams.ordinary_withdrawal_tax_rate
    )
    # Sims
    p.add_argument("--n-sims", type=int, default=ScenarioParams.n_sims)
    p.add_argument("--seed", type=int, default=ScenarioParams.seed)
    # Risk
    p.add_argument(
        "--risk-aversion", type=float, default=ScenarioParams.risk_aversion_gamma
    )
    # Output
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save-plots", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    return p


def run_cli(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)

    debt = DebtParams(
        balance=args.debt_balance,
        annual_rate=args.debt_rate,
        remaining_term_months=int(round(args.debt_term_years * 12)),
        min_payment_override=args.min_payment,
    )
    tax = TaxParams(
        interest_deductible_fraction=args.interest_deductible_fraction,
        marginal_income_tax_rate=args.marginal_tax,
        account_type=args.account_type,
        dividend_tax_rate=args.div_tax,
        capital_gains_tax_rate=args.cg_tax,
        ordinary_withdrawal_tax_rate=args.withdraw_tax,
    )
    invest = InvestParams(
        exp_return_annual=args.exp_return,
        vol_annual=args.vol,
        dividend_yield_annual=args.div_yield,
        stochastic=not args.deterministic,
    )
    scen = ScenarioParams(
        horizon_years=args.horizon_years,
        inflation_annual=args.inflation,
        monthly_surplus=args.monthly_surplus,
        n_sims=1 if args.deterministic else args.n_sims,
        seed=args.seed,
        risk_aversion_gamma=args.risk_aversion,
    )

    results = run_simulation(debt, invest, tax, scen)
    print_report(results, scen)

    if args.plot or args.save_plots:
        plot_rep_paths(
            results,
            scen,
            show=(args.plot and not args.no_show),
            save_path=args.save_plots,
        )


# -----------------------------
# Streamlit GUI
# -----------------------------


def is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit_app():
    import streamlit as st
    import pandas as pd
    import altair as alt

    # Streamlit "dark" background (works well in dark mode)
    BG = "#0E1117"
    FG = "#FAFAFA"
    GRID = "#2A2F3A"

    def dark_theme(chart: alt.Chart) -> alt.Chart:
        return (
            chart.properties(background=BG)
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor=FG,
                titleColor=FG,
                gridColor=GRID,
                domainColor=GRID,
                tickColor=GRID,
            )
            .configure_legend(labelColor=FG, titleColor=FG)
            .configure_title(color=FG)
        )

    # Distinct colors
    C_INV = "#1f77b4"  # invest
    C_PAY = "#ff7f0e"  # pay-down
    C_DIFF = "#2ca02c"  # difference

    HELP = {
        "debt_balance": "Outstanding principal today (e.g., remaining mortgage balance).",
        "debt_rate": "Annual percentage rate (APR). Used to compute monthly interest on the remaining balance.",
        "debt_term_years": "Years remaining on the amortization schedule (used to compute the minimum payment).",
        "min_payment_override": "If checked, you can enter your own required minimum payment.",
        "min_payment": "Required monthly payment for principal+interest (P&I).",
        "monthly_surplus": "Extra cash per month you can allocate either to extra principal payments or investing.",
        "horizon_years": "How long you want to compare outcomes (often until retirement or planned sale/refi).",
        "inflation": "Annual inflation rate to report results in 'real' (inflation-adjusted) dollars.",
        "exp_return": "Expected *nominal* annual return (arithmetic mean). Assumption, not a guarantee.",
        "vol": "Annual volatility. Higher volatility = wider distribution of outcomes.",
        "div_yield": "Annual dividend yield. Taxable accounts pay dividend taxes as they occur (simplified).",
        "stochastic": "If on, uses Monte Carlo random market paths; if off, uses a deterministic return path.",
        "n_sims": "Number of Monte Carlo simulations. More = smoother/stabler estimates but slower.",
        "seed": "Random seed (reproducibility). Change it to see variability at low simulation counts.",
        "account_type": (
            "Tax treatment:\n"
            "• taxable: dividends taxed yearly + capital gains taxed at end\n"
            "• traditional: taxed at withdrawal\n"
            "• roth: assumed tax-free"
        ),
        "interest_deduct": (
            "Effective fraction (0..1) of each $1 of interest that reduces taxable income.\n"
            "• 0.0 = no benefit (e.g., standard deduction)\n"
            "• 1.0 = fully deductible at your marginal rate\n"
            "Intermediate values = partial benefit."
        ),
        "marginal_tax": "Marginal income tax rate used for the interest-deduction benefit (simplified).",
        "div_tax": "Dividend tax rate (taxable accounts, simplified).",
        "cg_tax": "Capital gains tax rate at end of horizon (taxable accounts, simplified).",
        "withdraw_tax": "Tax rate applied to traditional-account withdrawals at end (simplified).",
        "risk_aversion": "Optional risk adjustment (CRRA certainty-equivalent) on *real* terminal wealth.",
        "run_button": "Click to recompute (prevents re-running on every tiny slider change).",
        "bands": "Bands are computed from an unbiased reservoir sample of simulations (keeps it fast at high sim counts).",
    }

    st.title("Pay Down Debt vs Invest Surplus")
    st.caption(
        "Educational model (simplified taxes/markets). "
        "Compares (A) pay down debt faster vs (B) invest surplus while paying debt on schedule."
    )

    with st.sidebar:
        st.header("Debt")
        debt_balance = st.number_input(
            "Current debt balance ($)",
            min_value=0.0,
            value=400000.0,
            step=1000.0,
            help=HELP["debt_balance"],
        )
        debt_rate = st.number_input(
            "Debt APR (e.g. 0.065 = 6.5%)",
            min_value=0.0,
            value=0.065,
            step=0.001,
            format="%.4f",
            help=HELP["debt_rate"],
        )
        debt_term_years = st.number_input(
            "Remaining term (years)",
            min_value=0.0,
            value=30.0,
            step=1.0,
            help=HELP["debt_term_years"],
        )
        min_payment_override = st.checkbox(
            "Override minimum payment?", value=False, help=HELP["min_payment_override"]
        )
        min_payment = None
        if min_payment_override:
            min_payment = st.number_input(
                "Minimum payment override ($/mo)",
                min_value=0.0,
                value=2500.0,
                step=50.0,
                help=HELP["min_payment"],
            )

        st.header("Cashflow")
        monthly_surplus = st.number_input(
            "Monthly surplus to allocate ($/mo)",
            min_value=0.0,
            value=1000.0,
            step=50.0,
            help=HELP["monthly_surplus"],
        )
        horizon_years = st.number_input(
            "Time horizon (years)",
            min_value=0.5,
            value=30.0,
            step=0.5,
            help=HELP["horizon_years"],
        )
        inflation = st.number_input(
            "Inflation (annual, e.g. 0.025)",
            min_value=0.0,
            value=0.025,
            step=0.001,
            format="%.4f",
            help=HELP["inflation"],
        )

        st.header("Investments")
        exp_return = st.number_input(
            "Expected return (annual, nominal)",
            value=0.07,
            step=0.005,
            format="%.4f",
            help=HELP["exp_return"],
        )
        vol = st.number_input(
            "Volatility (annual)",
            value=0.15,
            step=0.01,
            format="%.4f",
            help=HELP["vol"],
        )
        div_yield = st.number_input(
            "Dividend yield (annual)",
            value=0.015,
            step=0.001,
            format="%.4f",
            help=HELP["div_yield"],
        )
        stochastic = st.checkbox(
            "Use Monte Carlo (stochastic returns)", value=True, help=HELP["stochastic"]
        )

        # Default kept at 200 for speed
        n_sims = (
            st.slider(
                "Simulations",
                min_value=200,
                max_value=50000,
                value=200,
                step=200,
                help=HELP["n_sims"],
            )
            if stochastic
            else 1
        )
        seed = st.number_input(
            "Random seed", min_value=0, value=42, step=1, help=HELP["seed"]
        )

        st.header("Taxes (simplified)")
        account_type = st.selectbox(
            "Account type",
            ["taxable", "traditional", "roth"],
            index=0,
            help=HELP["account_type"],
        )
        interest_deduct_frac = st.number_input(
            "Effective interest deductible fraction (0..1)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help=HELP["interest_deduct"],
        )
        marginal_tax = st.number_input(
            "Marginal income tax rate",
            min_value=0.0,
            max_value=0.6,
            value=0.24,
            step=0.01,
            help=HELP["marginal_tax"],
        )
        div_tax = st.number_input(
            "Dividend tax rate",
            min_value=0.0,
            max_value=0.6,
            value=0.15,
            step=0.01,
            help=HELP["div_tax"],
        )
        cg_tax = st.number_input(
            "Capital gains tax rate",
            min_value=0.0,
            max_value=0.6,
            value=0.15,
            step=0.01,
            help=HELP["cg_tax"],
        )
        withdraw_tax = st.number_input(
            "Traditional withdrawal tax rate",
            min_value=0.0,
            max_value=0.6,
            value=0.24,
            step=0.01,
            help=HELP["withdraw_tax"],
        )

        st.header("Risk preference (optional)")
        gamma = st.number_input(
            "Risk aversion gamma (0 = risk-neutral)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help=HELP["risk_aversion"],
        )

        run = st.button("Run simulation", type="primary", help=HELP["run_button"])

    debt = DebtParams(
        balance=float(debt_balance),
        annual_rate=float(debt_rate),
        remaining_term_months=int(round(float(debt_term_years) * 12)),
        min_payment_override=float(min_payment) if min_payment_override else None,
    )
    tax = TaxParams(
        interest_deductible_fraction=float(interest_deduct_frac),
        marginal_income_tax_rate=float(marginal_tax),
        account_type=str(account_type),
        dividend_tax_rate=float(div_tax),
        capital_gains_tax_rate=float(cg_tax),
        ordinary_withdrawal_tax_rate=float(withdraw_tax),
    )
    invest = InvestParams(
        exp_return_annual=float(exp_return),
        vol_annual=float(vol),
        dividend_yield_annual=float(div_yield),
        stochastic=bool(stochastic),
    )
    scen = ScenarioParams(
        horizon_years=float(horizon_years),
        inflation_annual=float(inflation),
        monthly_surplus=float(monthly_surplus),
        n_sims=int(n_sims),
        seed=int(seed),
        risk_aversion_gamma=float(gamma),
    )

    @st.cache_data(show_spinner=False)
    def cached_run_simulation(debt_dict, invest_dict, tax_dict, scen_dict):
        return run_simulation(
            DebtParams(**debt_dict),
            InvestParams(**invest_dict),
            TaxParams(**tax_dict),
            ScenarioParams(**scen_dict),
            path_sample_size=2000,
        )

    if run or "results" not in st.session_state:
        with st.spinner("Running simulation..."):
            st.session_state["results"] = cached_run_simulation(
                asdict(debt), asdict(invest), asdict(tax), asdict(scen)
            )

    results = st.session_state["results"]

    diff_real = np.asarray(results["diff_real"])
    p_win_real = float(np.mean(diff_real > 0.0))
    sdr = summarize(diff_real)

    st.subheader("Key result (real, inflation-adjusted)")
    st.write(
        f"**P(Invest beats Pay-down)**: **{p_win_real*100:.1f}%**  \n"
        f"**Median (B-A)**: **{money(sdr['median'])}**  \n"
        f"**10th–90th percentile (B-A)**: **{money(sdr['p10'])}** to **{money(sdr['p90'])}**"
    )
    if "ce_diff_real" in results:
        st.write(
            f"**Risk-adjusted (certainty equivalent, real)**: CE(B) − CE(A) = **{money(float(results['ce_diff_real']))}**"
        )

    rep = results["rep_paths"]
    rep_pay_real = np.asarray(rep["pay_down"]["net_worth_real"])
    rep_inv_real = np.asarray(rep["invest"]["net_worth_real"])
    rep_pay_debt = np.asarray(rep["pay_down"]["debt_balance"])
    rep_inv_debt = np.asarray(rep["invest"]["debt_balance"])
    t_years = np.arange(len(rep_pay_real)) / 12.0

    # Compute band stats (if available)
    sample_pay = results.get("path_sample_paydown_real", None)
    sample_inv = results.get("path_sample_invest_real", None)
    k = int(results.get("path_sample_size", 0))

    has_bands = (
        isinstance(sample_pay, np.ndarray)
        and isinstance(sample_inv, np.ndarray)
        and k > 0
    )
    if has_bands:
        pay_p10, pay_p50, pay_p90 = np.percentile(sample_pay, [10, 50, 90], axis=0)
        inv_p10, inv_p50, inv_p90 = np.percentile(sample_inv, [10, 50, 90], axis=0)

        diff_sample = sample_inv - sample_pay
        d_p10, d_p50, d_p90 = np.percentile(diff_sample, [10, 50, 90], axis=0)

    # Helper: line chart with hover rule using wide-form df (columns in tooltip)
    def wide_line_with_hover(
        df_wide: pd.DataFrame,
        y_cols: List[str],
        colors: Dict[str, str],
        title: str,
        y_title: str,
    ):
        base = alt.Chart(df_wide).encode(x=alt.X("year:Q", title="Years"))
        nearest = alt.selection_point(
            nearest=True, on="mouseover", fields=["year"], empty=False
        )

        lines = (
            base.transform_fold(y_cols, as_=["series", "value"])
            .mark_line()
            .encode(
                y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format="$,.0f")),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=list(colors.keys()), range=list(colors.values())
                    ),
                    legend=alt.Legend(title=None),
                ),
            )
        )

        selectors = base.mark_point(opacity=0).add_params(nearest)
        points = lines.mark_point(size=40).encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        rule = (
            base.mark_rule(opacity=0.35)
            .encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                tooltip=[alt.Tooltip("year:Q", title="Year", format=".1f")]
                + [
                    alt.Tooltip(
                        c + ":Q", title=c.replace("_", " ").title(), format="$,.0f"
                    )
                    for c in y_cols
                ],
            )
            .transform_filter(nearest)
        )

        chart = alt.layer(lines, selectors, points, rule).properties(
            title=title, height=320
        )
        return dark_theme(chart)

    # Helper: band + median lines, with hover
    def band_chart_with_hover(df_band: pd.DataFrame, title: str):
        base = alt.Chart(df_band).encode(x=alt.X("year:Q", title="Years"))
        nearest = alt.selection_point(
            nearest=True, on="mouseover", fields=["year"], empty=False
        )

        invest_area = base.mark_area(opacity=0.18, color=C_INV).encode(
            y=alt.Y(
                "invest_p10:Q",
                title="Real net worth ($)",
                axis=alt.Axis(format="$,.0f"),
            ),
            y2="invest_p90:Q",
        )
        invest_med = base.mark_line(color=C_INV).encode(y="invest_median:Q")

        pay_area = base.mark_area(opacity=0.18, color=C_PAY).encode(
            y="paydown_p10:Q",
            y2="paydown_p90:Q",
        )
        pay_med = base.mark_line(color=C_PAY).encode(y="paydown_median:Q")

        rule = (
            base.mark_rule(opacity=0.35)
            .encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                tooltip=[
                    alt.Tooltip("year:Q", title="Year", format=".1f"),
                    alt.Tooltip(
                        "invest_median:Q", title="Invest (median)", format="$,.0f"
                    ),
                    alt.Tooltip("invest_p10:Q", title="Invest (p10)", format="$,.0f"),
                    alt.Tooltip("invest_p90:Q", title="Invest (p90)", format="$,.0f"),
                    alt.Tooltip(
                        "paydown_median:Q", title="Pay-down (median)", format="$,.0f"
                    ),
                    alt.Tooltip(
                        "paydown_p10:Q", title="Pay-down (p10)", format="$,.0f"
                    ),
                    alt.Tooltip(
                        "paydown_p90:Q", title="Pay-down (p90)", format="$,.0f"
                    ),
                ],
            )
            .add_params(nearest)
            .transform_filter(nearest)
        )

        chart = alt.layer(invest_area, pay_area, invest_med, pay_med, rule).properties(
            title=title, height=340
        )
        return dark_theme(chart)

    # Helper: difference chart (rep + band), hover
    def diff_chart_with_hover(df_diff: pd.DataFrame, title: str):
        base = alt.Chart(df_diff).encode(x=alt.X("year:Q", title="Years"))
        nearest = alt.selection_point(
            nearest=True, on="mouseover", fields=["year"], empty=False
        )

        zero = (
            alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(opacity=0.5).encode(y="y:Q")
        )

        layers = [zero]

        if {"diff_p10", "diff_p90"}.issubset(df_diff.columns):
            band = base.mark_area(opacity=0.18, color=C_DIFF).encode(
                y=alt.Y(
                    "diff_p10:Q",
                    title="Real difference ($)",
                    axis=alt.Axis(format="$,.0f"),
                ),
                y2="diff_p90:Q",
            )
            med = base.mark_line(color=C_DIFF, strokeDash=[6, 4]).encode(
                y="diff_median:Q"
            )
            layers += [band, med]
        else:
            # Still define axis title
            layers += [
                base.mark_line(opacity=0).encode(
                    y=alt.Y(
                        "diff_rep:Q",
                        title="Real difference ($)",
                        axis=alt.Axis(format="$,.0f"),
                    )
                )
            ]

        rep_line = base.mark_line(color=C_DIFF).encode(y="diff_rep:Q")
        layers.append(rep_line)

        tooltip_fields = [
            alt.Tooltip("year:Q", title="Year", format=".1f"),
            alt.Tooltip("diff_rep:Q", title="Rep diff", format="$,.0f"),
        ]
        if "diff_median" in df_diff.columns:
            tooltip_fields += [
                alt.Tooltip("diff_median:Q", title="Median diff", format="$,.0f"),
                alt.Tooltip("diff_p10:Q", title="p10 diff", format="$,.0f"),
                alt.Tooltip("diff_p90:Q", title="p90 diff", format="$,.0f"),
            ]

        rule = (
            base.mark_rule(opacity=0.35)
            .encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
                tooltip=tooltip_fields,
            )
            .add_params(nearest)
            .transform_filter(nearest)
        )
        layers.append(rule)

        chart = alt.layer(*layers).properties(title=title, height=320)
        return dark_theme(chart)

    # -----------------------------
    # Build dataframes for charts
    # -----------------------------

    df_rep_real = pd.DataFrame(
        {"year": t_years, "invest": rep_inv_real, "paydown": rep_pay_real}
    )
    df_rep_debt = pd.DataFrame(
        {"year": t_years, "invest_debt": rep_inv_debt, "paydown_debt": rep_pay_debt}
    )

    # 1) Representative real net worth
    st.subheader("1) Representative real net worth (single simulation)")
    st.altair_chart(
        wide_line_with_hover(
            df_rep_real,
            y_cols=["invest", "paydown"],
            colors={"invest": C_INV, "paydown": C_PAY},
            title="Real net worth (one market draw)",
            y_title="Real net worth ($)",
        ),
        use_container_width=True,
    )

    # 2) Median + band over time (sampled)
    st.subheader("2) Median + 10–90% band over time (sampled simulations)")
    st.caption(HELP["bands"])
    if has_bands:
        df_band = pd.DataFrame(
            {
                "year": t_years,
                "invest_median": inv_p50,
                "invest_p10": inv_p10,
                "invest_p90": inv_p90,
                "paydown_median": pay_p50,
                "paydown_p10": pay_p10,
                "paydown_p90": pay_p90,
            }
        )
        st.altair_chart(
            band_chart_with_hover(
                df_band, title=f"Median and 10–90% band (sample size = {k:,})"
            ),
            use_container_width=True,
        )
    else:
        st.info("Band chart unavailable (no sampled paths).")

    # 3) Debt decay plot
    st.subheader("3) Debt decay (representative single simulation)")
    # Debt is not "real dollars"—it's nominal debt balance—so use a separate axis format but keep tooltips.
    df_debt = df_rep_debt.copy()
    base_debt = alt.Chart(df_debt).encode(x=alt.X("year:Q", title="Years"))
    nearest_debt = alt.selection_point(
        nearest=True, on="mouseover", fields=["year"], empty=False
    )

    debt_lines = (
        base_debt.transform_fold(
            ["invest_debt", "paydown_debt"], as_=["series", "value"]
        )
        .mark_line()
        .encode(
            y=alt.Y("value:Q", title="Debt balance ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["invest_debt", "paydown_debt"],
                    range=[C_INV, C_PAY],
                ),
                legend=alt.Legend(title=None),
            ),
        )
    )
    debt_selectors = base_debt.mark_point(opacity=0).add_params(nearest_debt)
    debt_points = debt_lines.mark_point(size=40).encode(
        opacity=alt.condition(nearest_debt, alt.value(1), alt.value(0))
    )
    debt_rule = (
        base_debt.mark_rule(opacity=0.35)
        .encode(
            opacity=alt.condition(nearest_debt, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("year:Q", title="Year", format=".1f"),
                alt.Tooltip("invest_debt:Q", title="Invest debt", format="$,.0f"),
                alt.Tooltip("paydown_debt:Q", title="Pay-down debt", format="$,.0f"),
            ],
        )
        .transform_filter(nearest_debt)
    )
    debt_chart = alt.layer(
        debt_lines, debt_selectors, debt_points, debt_rule
    ).properties(title="Debt balance over time (one market draw)", height=300)
    st.altair_chart(dark_theme(debt_chart), use_container_width=True)

    # 4) Difference chart (rep + median band)
    st.subheader("4) Difference: Invest − Pay-down (rep + median band)")
    df_diff = pd.DataFrame({"year": t_years, "diff_rep": rep_inv_real - rep_pay_real})
    if has_bands:
        df_diff["diff_median"] = d_p50
        df_diff["diff_p10"] = d_p10
        df_diff["diff_p90"] = d_p90

    st.altair_chart(
        diff_chart_with_hover(
            df_diff, title="Invest − Pay-down (positive means invest is ahead)"
        ),
        use_container_width=True,
    )

    # 5) Histogram
    st.subheader("5) Distribution of terminal outcomes (real, Invest − Pay-down)")
    df_hist = pd.DataFrame({"diff_real": diff_real})
    hist = (
        alt.Chart(df_hist)
        .mark_bar(color=C_DIFF, opacity=0.9)
        .encode(
            x=alt.X(
                "diff_real:Q",
                bin=alt.Bin(maxbins=60),
                title="Terminal real difference ($)",
            ),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
        .properties(title="Monte Carlo distribution", height=260)
    )
    st.altair_chart(dark_theme(hist), use_container_width=True)


# -----------------------------
# Entry point
# -----------------------------


def run_streamlit():
    """
    Console entrypoint: `debt-vs-invest`
    Runs Streamlit pointing at this file.
    """
    import os
    import sys
    from streamlit.web import cli as stcli

    # Run: streamlit run <this_file>
    this_file = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", this_file]
    sys.exit(stcli.main())


def main_cli():
    """
    Console entrypoint: `debt-vs-invest-cli`
    Provides the optional CLI mode.
    """
    run_cli()


if __name__ == "__main__":
    # When invoked directly, default to Streamlit UX.
    run_streamlit()
