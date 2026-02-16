# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-16

### Added
- `fixed_income.instruments.FixedRateBond` - Bond pricing, YTM, DV01, convexity, CPN01, zero-curve PV
- `fixed_income.instruments.FedFundsFutures` - FOMC rate extraction (bootstrap and least-squares)
- `fixed_income.curves.ZeroCurve` - Abstract base class for yield curves
- `fixed_income.curves.DummyZeroCurve` - Flat-rate curve for testing
- `fixed_income.curves.NelsonSiegelSvenssonSpline` - NS/NSS yield curve model with fitting
- `fixed_income.curves.FFERCurve` - Fed Funds Effective Rate front-end curve
- `fixed_income.calibration.CalibrateNelsonSiegelSvensson` - Historical curve calibration
- `fixed_income.dates` - Business day adjustment, IMM dates, CBOT codes, day counts, date conversion
- `fixed_income.market_data` - Bloomberg Excel data reader
- `pyproject.toml` with declared dependencies and dynamic versioning
- `__version__` attribute for runtime version access

### Fixed
- `calibration/__init__.py` filename typo (was `__init_.py`, not importable)
- `dates/utils.py` NameError (`datetime.datetime` vs `dt.datetime`)
- Bond coupon schedule using wrong day convention (was `Modified Following`, now `None` with `eom=True`)
- `ZeroCurve` subclass method signatures missing `comp` parameter
