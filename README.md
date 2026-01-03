# Debt vs Invest (Streamlit)

A Streamlit app that helps you compare two common "what should I do with my extra cash?" strategies:

- **Pay-down-first:** use extra monthly cash to pay down debt faster (e.g., mortgage principal), then invest the freed cash once the debt is gone.
- **Invest-first:** invest the extra monthly cash while paying the debt on schedule, then invest the freed cash after the debt is naturally paid off.

The tool runs a **Monte Carlo simulation** (optional deterministic mode) with inflation and simplified tax assumptions, and visualizes outcomes with interactive, hoverable charts.

> **Disclaimer:** This project is for educational purposes only. It uses simplified assumptions and is not financial, legal, or tax advice.

---

## Features

- **Interactive Streamlit GUI** (primary use case)
- **Monte Carlo market simulation** (expected return + volatility)
- **Inflation-adjusted ("real") results**
- **Simplified taxes**
  - taxable: dividend taxes annually + capital gains tax at horizon
  - traditional: taxed at withdrawal at horizon
  - roth: assumed tax-free
- **Simplified interest deductibility** (user supplies an "effective deductible fraction")
- **Interactive charts with hover tooltips**
  1. Representative real net worth (single simulation)
  2. Median + 10-90% band over time (sampled)
  3. Debt decay plot (representative)
  4. Difference over time: Invest - Pay-down (rep + median band)
  5. Terminal outcome histogram (real)

---

## Installation

### Requirements
- Python **3.10+**
- Works on macOS / Linux / Windows

### Option A (recommended): Install via `pyproject.toml` (editable)
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -U pip
pip install -e .
```

This installs dependencies and enables running the app as a command (if you include the recommended entrypoint in `pyproject.toml`).

### Option B: Install dependencies only (no packaging)
If you'd rather not install as a package, you can still run Streamlit directly:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install numpy pandas altair streamlit matplotlib
```

---

## Running the app (Streamlit)

### If installed as a package (recommended entrypoint)
```bash
debt-vs-invest
```

### Or run Streamlit directly
```bash
streamlit run debt_vs_invest/app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) and open the app in your browser.

---

## Quick start guide

1. **Enter your debt details**
   - Current balance
   - APR
   - Remaining term (years)
   - (Optional) override your minimum payment if you want

2. **Set your monthly surplus**
   - This is the extra cash you could either invest or use for extra principal payments.

3. **Set market assumptions**
   - Expected return (nominal)
   - Volatility
   - Dividend yield
   - Number of simulations (defaults low for speed)

4. **Set inflation**
   - Used to convert outcomes to "real dollars" so you can compare purchasing power.

5. **Set simplified tax assumptions**
   - Account type (taxable / traditional / roth)
   - Dividend tax rate / cap gains tax rate / withdrawal tax rate
   - Interest deduction fraction (see below)

6. Click **Run simulation**

---

## How to interpret the outputs

### "Invest beats Pay-down X%"
This is the fraction of Monte Carlo simulations where the **terminal (end-of-horizon)** real net worth is higher for the invest-first strategy.

### Representative path vs median band
- The **representative** line is a *single simulation* (one random market path).
- The **median + 10-90% band** summarizes a sample of simulations over time and better reflects typical outcomes and uncertainty.

### Difference chart
This shows **(Invest - Pay-down)** over time:
- positive = investing is ahead
- negative = paying down is ahead
- includes both a representative path and (when available) a median band

---

## Modeling assumptions (important)

This is intentionally a simplified model designed for intuition, not for perfect accounting.

### Debt
- Modeled as a **fixed-rate amortizing loan**
- Minimum payment is computed from balance / APR / remaining term unless overridden
- Pay-down-first strategy applies **(minimum payment + monthly surplus)** toward debt until paid off

### Investments
- Monthly returns are modeled as lognormal consistent with the chosen annual expected return and volatility
- Dividend yield is applied monthly as a simplified dividend model

### Taxes (simplified)
- **Taxable accounts**
  - dividends taxed as they occur
  - capital gains tax applied at the end of the horizon
- **Traditional**
  - taxes applied at withdrawal at the end of the horizon
- **Roth**
  - assumed tax-free

> Real-world taxes are more complex (tax brackets, loss harvesting, deduction caps, AMT, state taxes, etc.). This model simplifies these by design.

### Interest deductibility (simplified)
The "interest deductible fraction" is an *effective* knob that captures how much of your interest actually reduces your taxes.

Examples:
- `0.0` -> no benefit (e.g., you take the standard deduction)
- `1.0` -> fully deductible interest at your marginal rate
- `0.3` -> partial benefit (e.g., some of your interest is effectively deductible)

---

## Performance tips

- The default simulation count is set low (e.g., 200) for responsiveness.
- Increase simulations for smoother percentile bands and more stable win-rate estimates.
- If you crank simulations very high, the UI may feel slower; that's expected.

---

## CLI (optional)

If you included the CLI entrypoint, you can run:

```bash
debt-vs-invest-cli --help
```

Example:

```bash
debt-vs-invest-cli \
  --debt-balance 400000 \
  --debt-rate 0.065 \
  --debt-term-years 30 \
  --monthly-surplus 1000 \
  --horizon-years 30 \
  --inflation 0.025 \
  --exp-return 0.07 \
  --vol 0.15 \
  --div-yield 0.015 \
  --account-type taxable \
  --n-sims 5000 \
  --plot
```

> The Streamlit UI is the intended primary interface.

---

## Suggested repo contents

A minimal repo typically includes:

- `pyproject.toml`
- `debt_vs_invest/app.py`
- `README.md`
- `LICENSE`
- `.gitignore`

If you want Streamlit itself to default to dark theme, you can add:

**`.streamlit/config.toml`**
```toml
[theme]
base="dark"
```

---

## License

MIT (recommended). See `LICENSE`.

---

## Acknowledgements

This project was inspired by common personal finance questions around whether to:
- accelerate debt payoff (e.g., mortgage principal), or
- invest surplus cash (e.g., broad index funds)

If you extend the model (refinance, variable rates, contribution limits, state taxes, etc.), PRs are welcome.
