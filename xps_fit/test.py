from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import optuna


# Edit this block directly or override the values with command-line arguments.
FIT_CONFIG = {
    # Only data inside this range are used for fitting.
    "fit_range": [776.0, 784.0],
    "n_peaks": 2,
    # Leave empty to use Peak 1, Peak 2, ...
    "peak_labels": [],
    # One [minimum, maximum] range per peak. Leave empty to search the full x range.
    # Example for two peaks: [[779.0, 780.5], [781.0, 782.5]]
    "center_ranges": [],
    "peak_width_range": [0.1, 3.0],  # Gaussian sigma, in x-axis units (usually eV)
    "peak_amplitude_fraction_range": [0.0, 1.5],
    "minimum_peak_separation": 0.1,
    # Values above zero give the high-intensity peak region more influence.
    "high_intensity_weight": 3.0,
    "n_trials": 2000,
    "seed": 7,
    "timeout": None,
    "n_startup_trials": 100,
    "spectrum_title": "XPS peak fitting",
}


@dataclass(frozen=True)
class PeakFit:
    label: str
    amplitude: float
    center: float
    sigma: float
    area: float
    area_ratio: float
    values: np.ndarray


@dataclass(frozen=True)
class FitResult:
    x: np.ndarray
    y: np.ndarray
    background: np.ndarray
    background_params: tuple[float, float, float, float]
    peaks: tuple[PeakFit, ...]
    fitted: np.ndarray
    normalized_rmse: float
    rmse: float
    r2: float


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 16,
            "axes.linewidth": 1.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def sigmoid_background(
    x: np.ndarray,
    offset: float,
    amplitude: float,
    center: float,
    width: float,
) -> np.ndarray:
    exponent = np.clip(-(x - center) / width, -700.0, 700.0)
    return offset + amplitude / (1.0 + np.exp(exponent))


def sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    order = np.argsort(x)
    return x[order], y[order]


def parse_peak_labels(text: str | None, n_peaks: int) -> list[str]:
    if text:
        labels = [item.strip() for item in text.split(",") if item.strip()]
    else:
        labels = [str(item).strip() for item in FIT_CONFIG["peak_labels"] if str(item).strip()]
    if not labels:
        return [f"Peak {index + 1}" for index in range(n_peaks)]
    if len(labels) != n_peaks:
        raise ValueError(f"Expected {n_peaks} peak labels, received {len(labels)}.")
    return labels


def parse_fit_range(
    text: str | None,
    data_min: float,
    data_max: float,
) -> tuple[float, float]:
    if text:
        values = text.strip().split(":")
        if len(values) != 2:
            raise ValueError("fit-range must use min:max, for example 775:785.")
        lower, upper = sorted((float(values[0]), float(values[1])))
    else:
        configured = FIT_CONFIG["fit_range"]
        if len(configured) != 2:
            raise ValueError("FIT_CONFIG['fit_range'] must contain [minimum, maximum].")
        lower, upper = sorted((float(configured[0]), float(configured[1])))

    if lower < data_min or upper > data_max or lower == upper:
        raise ValueError(
            f"Fit range [{lower}, {upper}] must lie inside the data range "
            f"[{data_min}, {data_max}]."
        )
    return lower, upper


def select_fit_region(
    x: np.ndarray,
    y: np.ndarray,
    fit_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = fit_range
    mask = (x >= lower) & (x <= upper)
    selected_x = x[mask]
    selected_y = y[mask]
    if len(selected_x) < 5:
        raise ValueError(
            f"Only {len(selected_x)} data points fall inside fit range [{lower}, {upper}]."
        )
    return selected_x, selected_y


def parse_center_ranges(
    text: str | None,
    n_peaks: int,
    x_min: float,
    x_max: float,
) -> list[tuple[float, float]]:
    if text:
        ranges: list[Sequence[float]] = []
        for item in text.split(","):
            values = item.strip().split(":")
            if len(values) != 2:
                raise ValueError(
                    "Center ranges must use min:max pairs, for example 779:780.5,781:782.5."
                )
            ranges.append([float(values[0]), float(values[1])])
    else:
        ranges = FIT_CONFIG["center_ranges"]

    if not ranges:
        return [(x_min, x_max) for _ in range(n_peaks)]
    if len(ranges) != n_peaks:
        raise ValueError(f"Expected {n_peaks} center ranges, received {len(ranges)}.")

    parsed: list[tuple[float, float]] = []
    for index, values in enumerate(ranges, start=1):
        if len(values) != 2:
            raise ValueError(f"Center range {index} must contain exactly two values.")
        lower, upper = sorted((float(values[0]), float(values[1])))
        if lower < x_min or upper > x_max or lower == upper:
            raise ValueError(
                f"Center range {index} [{lower}, {upper}] must lie inside "
                f"the data range [{x_min}, {x_max}]."
            )
        parsed.append((lower, upper))
    return parsed


def model_components(
    x: np.ndarray,
    background_params: Sequence[float],
    peak_params: Sequence[Sequence[float]],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    background = sigmoid_background(x, *background_params)
    peaks = [gaussian(x, *params) for params in peak_params]
    fitted = background + np.sum(peaks, axis=0)
    return background, peaks, fitted


def fit_xps_peaks(
    x: np.ndarray,
    y: np.ndarray,
    n_peaks: int,
    labels: Sequence[str],
    center_ranges: Sequence[tuple[float, float]],
    n_trials: int,
    seed: int,
    timeout: float | None,
    minimum_peak_separation: float,
    high_intensity_weight: float,
) -> tuple[FitResult, optuna.Study]:
    x, y = sort_xy(x, y)
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1.")
    if len(labels) != n_peaks or len(center_ranges) != n_peaks:
        raise ValueError("Peak labels and center ranges must match n_peaks.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if minimum_peak_separation < 0.0:
        raise ValueError("minimum_peak_separation cannot be negative.")
    if high_intensity_weight < 0.0:
        raise ValueError("high_intensity_weight cannot be negative.")

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    intensity_scale = max(float(np.ptp(y)), 1.0)
    point_weights = 1.0 + high_intensity_weight * (y - y_min) / intensity_scale
    x_span = float(np.ptp(x))
    if x_span <= 0.0:
        raise ValueError("The x-axis must contain more than one unique value.")
    unique_x = np.unique(x)
    minimum_x_step = float(np.min(np.diff(unique_x))) if len(unique_x) > 1 else x_span

    amplitude_bounds = [float(value) for value in FIT_CONFIG["peak_amplitude_fraction_range"]]
    width_bounds = [float(value) for value in FIT_CONFIG["peak_width_range"]]
    if amplitude_bounds[0] < 0.0 or amplitude_bounds[0] > amplitude_bounds[1]:
        raise ValueError("Invalid peak_amplitude_fraction_range.")
    if width_bounds[0] <= 0.0 or width_bounds[0] >= width_bounds[1]:
        raise ValueError("Invalid peak_width_range.")

    def parameters_from_trial(
        trial: optuna.Trial,
    ) -> tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]:
        background_params = (
            trial.suggest_float("background_offset", y_min - 0.25 * intensity_scale, y_max),
            intensity_scale * trial.suggest_float("background_amplitude_fraction", -1.5, 1.5),
            trial.suggest_float("background_center", float(x[0]), float(x[-1])),
            trial.suggest_float(
                "background_width",
                max(minimum_x_step, 1e-6),
                max(x_span, minimum_x_step * 1.01),
                log=True,
            ),
        )
        peak_params = []
        for index, center_range in enumerate(center_ranges, start=1):
            peak_params.append(
                (
                    intensity_scale
                    * trial.suggest_float(
                        f"peak_{index}_amplitude_fraction", *amplitude_bounds
                    ),
                    trial.suggest_float(f"peak_{index}_center", *center_range),
                    trial.suggest_float(f"peak_{index}_sigma", *width_bounds),
                )
            )
        return background_params, peak_params

    def objective(trial: optuna.Trial) -> float:
        background_params, peak_params = parameters_from_trial(trial)
        centers = np.sort([params[1] for params in peak_params])
        if len(centers) > 1:
            closest_distance = float(np.min(np.diff(centers)))
            if closest_distance < minimum_peak_separation:
                return 1_000.0 + (minimum_peak_separation - closest_distance) / x_span

        _, _, fitted = model_components(x, background_params, peak_params)
        if not np.all(np.isfinite(fitted)):
            return float("inf")
        normalized_residual = (y - fitted) / intensity_scale
        return float(
            np.sqrt(np.sum(point_weights * normalized_residual**2) / np.sum(point_weights))
        )

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(int(FIT_CONFIG["n_startup_trials"]), n_trials),
        multivariate=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    initial_trial = {
        "background_offset": y_min,
        "background_amplitude_fraction": (float(y[-1]) - float(y[0])) / intensity_scale,
        "background_center": float(np.mean(x)),
        "background_width": max(0.1 * x_span, minimum_x_step),
    }
    for index, center_range in enumerate(center_ranges, start=1):
        initial_trial[f"peak_{index}_amplitude_fraction"] = min(1.0, 1.0 / n_peaks)
        initial_trial[f"peak_{index}_center"] = float(np.mean(center_range))
        initial_trial[f"peak_{index}_sigma"] = float(np.mean(width_bounds))
    study.enqueue_trial(initial_trial)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    background_params, raw_peak_params = parameters_from_trial(study.best_trial)
    ordered = sorted(
        zip(labels, raw_peak_params, strict=True),
        key=lambda item: item[1][1],
    )
    ordered_labels = [item[0] for item in ordered]
    ordered_params = [item[1] for item in ordered]
    background, peak_values, fitted = model_components(x, background_params, ordered_params)

    areas = np.array([float(np.trapz(values, x)) for values in peak_values])
    total_area = float(np.sum(areas))
    if total_area <= 0.0:
        raise ValueError("The total fitted peak area is not positive.")
    ratios = areas / total_area

    peaks = tuple(
        PeakFit(
            label=label,
            amplitude=float(params[0]),
            center=float(params[1]),
            sigma=float(params[2]),
            area=float(area),
            area_ratio=float(ratio),
            values=values,
        )
        for label, params, area, ratio, values in zip(
            ordered_labels,
            ordered_params,
            areas,
            ratios,
            peak_values,
            strict=True,
        )
    )

    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    result = FitResult(
        x=x,
        y=y,
        background=background,
        background_params=tuple(float(value) for value in background_params),
        peaks=peaks,
        fitted=fitted,
        normalized_rmse=rmse / intensity_scale,
        rmse=rmse,
        r2=r2,
    )
    return result, study


def plot_fit(
    result: FitResult,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    fit_range: tuple[float, float],
    title: str,
    reverse_x: bool,
    output: Path | None,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = plt.colormaps["tab10"].resampled(max(len(result.peaks), 1))

    raw_x, raw_y = sort_xy(raw_x, raw_y)
    ax.plot(raw_x, raw_y, "o", ms=3.5, color="#F4A261", alpha=0.6, label="Raw data")
    ax.axvspan(
        fit_range[0],
        fit_range[1],
        color="gray",
        alpha=0.07,
        label="Fit region",
    )
    ax.plot(result.x, result.fitted, "-", lw=2.2, color="black", label=f"Fit ($R^2$={result.r2:.4f})")
    ax.plot(result.x, result.background, ":", lw=2, color="gray", label="Background")

    for index, peak in enumerate(result.peaks):
        color = colors(index)
        ax.plot(
            result.x,
            result.background + peak.values,
            "--",
            lw=2,
            color=color,
            label=f"{peak.label} ({peak.center:.2f})",
        )
        ax.fill_between(
            result.x,
            result.background,
            result.background + peak.values,
            color=color,
            alpha=0.28,
        )

    x_margin = 0.02 * float(np.ptp(raw_x))
    if reverse_x:
        ax.set_xlim(float(np.max(raw_x)) + x_margin, float(np.min(raw_x)) - x_margin)
    else:
        ax.set_xlim(float(np.min(raw_x)) - x_margin, float(np.max(raw_x)) + x_margin)
    y_span = max(float(np.ptp(raw_y)), 1.0)
    ax.set_ylim(float(np.min(raw_y)) - 0.08 * y_span, float(np.max(raw_y)) + 0.15 * y_span)
    ax.set_title(title, fontsize=19)
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend(loc="best", frameon=False, fontsize=9)
    ratio_text = "\n".join(f"{peak.label}: {peak.area_ratio:.1%}" for peak in result.peaks)
    ax.text(0.98, 0.97, ratio_text, transform=ax.transAxes, ha="right", va="top", fontsize=11)
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_components(result: FitResult, output: Path) -> None:
    columns = [result.x, result.y, result.background]
    headers = ["binding_energy_eV", "intensity", "background"]
    for index, peak in enumerate(result.peaks, start=1):
        columns.append(peak.values)
        headers.append(f"peak_{index}")
    columns.append(result.fitted)
    headers.append("fitted")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )


def plot_optimization_history(study: optuna.Study, output: Path) -> None:
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise ValueError("No completed Optuna trials are available for plotting.")
    trial_numbers = np.array([trial.number for trial in completed], dtype=int)
    values = np.array([float(trial.value) for trial in completed], dtype=float)
    best_so_far = np.minimum.accumulate(values)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(trial_numbers, values, s=12, alpha=0.3, label="Trial")
    ax.plot(trial_numbers, best_so_far, color="black", lw=2, label="Best so far")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Weighted normalized RMSE")
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary(result: FitResult, study: optuna.Study, output: Path) -> None:
    offset, amplitude, center, width = result.background_params
    summary = {
        "best_trial": study.best_trial.number,
        "n_peaks": len(result.peaks),
        "optuna_weighted_objective": study.best_value,
        "normalized_rmse": result.normalized_rmse,
        "rmse": result.rmse,
        "r2": result.r2,
        "background": {
            "offset": offset,
            "amplitude": amplitude,
            "center": center,
            "width": width,
        },
        "peaks": [
            {
                "label": peak.label,
                "amplitude": peak.amplitude,
                "center": peak.center,
                "sigma": peak.sigma,
                "fwhm": 2.354820045 * peak.sigma,
                "area": peak.area,
                "area_ratio": peak.area_ratio,
            }
            for peak in result.peaks
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_xy_csv(
    input_path: Path,
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {input_path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        name_lookup = {name.strip().lower(): name for name in reader.fieldnames}
        actual_x = name_lookup.get(x_column.lower())
        actual_y = name_lookup.get(y_column.lower())
        if actual_x is None or actual_y is None:
            raise ValueError(
                f"Requested columns '{x_column}' and '{y_column}' were not found. "
                f"Available columns: {', '.join(fieldnames)}"
            )

        x_values: list[float] = []
        y_values: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                x_value = float(row[actual_x])
                y_value = float(row[actual_y])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-numeric data in {input_path} at row {row_number}."
                ) from exc
            if np.isfinite(x_value) and np.isfinite(y_value):
                x_values.append(x_value)
                y_values.append(y_value)

    if len(x_values) < 3:
        raise ValueError("At least three finite x/y data points are required.")
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit an arbitrary number of Gaussian peaks to XPS data using Optuna."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).with_name("co_2p_components.csv")),
        help="Input CSV path.",
    )
    parser.add_argument("--x-column", default="binding_energy_eV")
    parser.add_argument("--y-column", default="intensity")
    parser.add_argument(
        "--fit-range",
        default=None,
        help="Energy range used for fitting, written as min:max, for example 775:785.",
    )
    parser.add_argument("--n-peaks", type=int, default=FIT_CONFIG["n_peaks"])
    parser.add_argument(
        "--peak-labels",
        default=None,
        help="Comma-separated labels, for example Co2+,Co3+.",
    )
    parser.add_argument(
        "--center-ranges",
        default=None,
        help="Comma-separated min:max ranges, for example 779:780.5,781:782.5.",
    )
    parser.add_argument(
        "--minimum-peak-separation",
        type=float,
        default=FIT_CONFIG["minimum_peak_separation"],
    )
    parser.add_argument(
        "--high-intensity-weight",
        type=float,
        default=FIT_CONFIG["high_intensity_weight"],
        help="Extra weight applied to high-intensity data points in the Optuna objective.",
    )
    parser.add_argument("--title", default=FIT_CONFIG["spectrum_title"])
    parser.add_argument("--ascending-x", action="store_true", help="Do not reverse the XPS x-axis.")
    parser.add_argument("--n-trials", type=int, default=FIT_CONFIG["n_trials"])
    parser.add_argument("--seed", type=int, default=FIT_CONFIG["seed"])
    parser.add_argument("--timeout", type=float, default=FIT_CONFIG["timeout"])
    parser.add_argument("--output", default="xps_peak_fit.png")
    parser.add_argument("--components", default="xps_peak_components.csv")
    parser.add_argument("--trials-output", default="xps_optuna_trials.csv")
    parser.add_argument("--history-output", default="xps_optuna_history.png")
    parser.add_argument("--summary-output", default="xps_fit_summary.json")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot()

    input_path = Path(args.input)
    raw_x, raw_y = load_xy_csv(input_path, args.x_column, args.y_column)
    data_min, data_max = sorted((float(np.min(raw_x)), float(np.max(raw_x))))
    fit_range = parse_fit_range(args.fit_range, data_min, data_max)
    x, y = select_fit_region(raw_x, raw_y, fit_range)
    x_min, x_max = sorted((float(np.min(x)), float(np.max(x))))
    labels = parse_peak_labels(args.peak_labels, args.n_peaks)
    center_ranges = parse_center_ranges(
        args.center_ranges,
        args.n_peaks,
        x_min,
        x_max,
    )
    print(f"Input: {input_path}")
    print(f"Fit range: {fit_range[0]:.4f} to {fit_range[1]:.4f} ({len(x)} points)")
    print(f"Fitting {args.n_peaks} peaks: {', '.join(labels)}")
    print(f"Center ranges: {center_ranges}")

    result, study = fit_xps_peaks(
        x=x,
        y=y,
        n_peaks=args.n_peaks,
        labels=labels,
        center_ranges=center_ranges,
        n_trials=args.n_trials,
        seed=args.seed,
        timeout=args.timeout,
        minimum_peak_separation=args.minimum_peak_separation,
        high_intensity_weight=args.high_intensity_weight,
    )

    output = Path(args.output) if args.output else None
    components = Path(args.components) if args.components else None
    trials_output = Path(args.trials_output) if args.trials_output else None
    history_output = Path(args.history_output) if args.history_output else None
    summary_output = Path(args.summary_output) if args.summary_output else None
    plot_fit(
        result,
        raw_x,
        raw_y,
        fit_range,
        args.title,
        not args.ascending_x,
        output,
        args.show,
    )
    if components is not None:
        write_components(result, components)
    if trials_output is not None:
        trials_output.parent.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(trials_output, index=False)
    if history_output is not None:
        plot_optimization_history(study, history_output)
    if summary_output is not None:
        write_summary(result, study, summary_output)

    print(f"Best Optuna trial: {study.best_trial.number}")
    for peak in result.peaks:
        print(
            f"{peak.label}: center={peak.center:.4f}, sigma={peak.sigma:.4f}, "
            f"FWHM={2.354820045 * peak.sigma:.4f}, area ratio={peak.area_ratio:.4f}"
        )
    print(f"Normalized RMSE: {result.normalized_rmse:.6f}")
    print(f"RMSE: {result.rmse:.6f}")
    print(f"R2: {result.r2:.6f}")


if __name__ == "__main__":
    main()
