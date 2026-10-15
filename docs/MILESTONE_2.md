# Milestone 2: environment and data audit

**Status: partially complete.** Environment verification, training-data exploration and calendar auditing are implemented. Volume adjustment semantics remain unresolved. Current task status is maintained in [PROGRESS.md](PROGRESS.md); milestone completion criteria are defined in [PLAN.md](PLAN.md).

## Environment and data evidence
The foundation environment uses Python 3.12. The historical foundation and dataset-stage environments are recorded in `requirements-foundation-lock.txt` and `requirements-milestone2-lock.txt`; current installation instructions are in the [README](../README.md#installation).

Frozen Yahoo snapshots contain NVDA, SPY and XLK daily observations for 2015–2025. The [acquisition audit](data_verification.json) records snapshot checksums and financial-data checks. These administrative checks cover the full snapshot and do not constitute forecast evaluation.

The [calendar audit](calendar_audit_2026-10-12.json) records 2,766 expected sessions per instrument, with no missing or unexpected dates, duplicates or ordering defects. NVDA uses the NASDAQ calendar; SPY and XLK use NYSE regular-equity sessions as a documented proxy for NYSE Arca. This is a calendar-library comparison, not exchange certification. The 2025 calendar check inspects dates only.

## Training-data exploration
[Notebook 01](../notebooks/01_data_understanding.ipynb) loads the configuration and frozen snapshots, converts timestamps to exchange-local dates, validates the data and selects training rows. Each series contains 2,014 training sessions from 2015-01-02 to 2022-12-30. NVDA's first split-adjusted Close is 0.50325 USD; this is not the original quoted price.

The comparative chart rebases each Close series to 100 on its first training date. This is a descriptive transformation, not model scaling or a dividend-reinvested total-return series. NVDA's growth motivates later consideration of distribution change; it does not demonstrate predictability.

Target definitions, adjustment limitations, session conventions and temporal boundaries are specified in [DATA_CONVENTIONS.md](DATA_CONVENTIONS.md). The rationale for selecting split-adjusted Close is recorded in [DECISIONS.md](DECISIONS.md).

## Verification and reproduction
Historical foundation verification comprised six passing tests, matching snapshot checksums, successful notebook 01 execution and consistent installed dependencies. The subsequent dataset/naive implementation run comprised 12 passing tests, including calendar exceptions. These are historical suite totals, not counts attributable solely to this milestone; the current verification record is maintained in [PROGRESS.md](PROGRESS.md).

Follow the [installation and data prerequisites](../README.md#installation), run `python scripts/check_snapshot.py`, and execute notebook 01 for training-only exploration. `scripts/audit_calendar.py` writes the fixed destination `docs/calendar_audit_2026-10-12.json` and refuses to overwrite it. A new audit requires a separately versioned destination and a recorded reason; preserve existing evidence.

## Outstanding work
Volume features remain excluded until the vendor's adjustment methodology is independently corroborated. Retrospective price adjustment and the calendar proxy remain documented limitations.

Dataset construction is described in [Milestone 3](MILESTONE_3.md), and forecast results are described in [Milestone 4](MILESTONE_4.md).
