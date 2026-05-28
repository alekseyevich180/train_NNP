from __future__ import annotations

from ase.calculators.calculator import Calculator
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


def parse_calc_mode(calc_mode_name: str) -> EstimatorCalcMode:
    try:
        return EstimatorCalcMode[calc_mode_name]
    except KeyError as exc:
        valid_names = ", ".join(mode.name for mode in EstimatorCalcMode)
        raise ValueError(f"Unknown calc mode '{calc_mode_name}'. Valid modes: {valid_names}") from exc


def build_pfp_calculator(calc_mode_name: str) -> Calculator:
    estimator = Estimator(calc_mode=parse_calc_mode(calc_mode_name))
    return ASECalculator(estimator)

