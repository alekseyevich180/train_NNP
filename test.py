from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FitResult:
    params: np.ndarray
    covariance: np.ndarray
    x: np.ndarray
    y: np.ndarray
    background: np.ndarray
    co2: np.ndarray
    co3: np.ndarray
    fitted: np.ndarray
    co2_ratio: float
    co3_ratio: float
    r2: float


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.linewidth": 1.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def gaussian(x: np.ndarray, amp: float, cen: float, wid: float) -> np.ndarray:
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))


def sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    order = np.argsort(x)
    return x[order], y[order]


def adaptive_background(
    x: np.ndarray,
    y_data: np.ndarray,
    x1_pos: float,
    x2_pos: float,
    bg_a: float,
    bg_b: float,
    bg_c: float,
) -> np.ndarray:
    lower = x1_pos - 3.3
    upper = x2_pos + 5.0
    transition_width = 0.7

    bg_sigmoid = bg_a / (1 + np.exp(-(x - bg_b) / bg_c))
    spline = interp1d(x, y_data, kind="cubic", bounds_error=False, fill_value="extrapolate")
    bg_spline = spline(x)

    w_left = 1 / (1 + np.exp(-(x - lower) / transition_width))
    w_right = 1 / (1 + np.exp((x - upper) / transition_width))
    weight = w_left * w_right
    return weight * bg_sigmoid + (1 - weight) * bg_spline


def make_model(y_data: np.ndarray):
    def model(
        x: np.ndarray,
        bg_a: float,
        bg_b: float,
        bg_c: float,
        a1: float,
        c1: float,
        w1: float,
        a2: float,
        c2: float,
        w2: float,
    ) -> np.ndarray:
        bg = adaptive_background(x, y_data, c1, c2, bg_a, bg_b, bg_c)
        return bg + gaussian(x, a1, c1, w1) + gaussian(x, a2, c2, w2)

    return model


def fit_co_2p(x: np.ndarray, y: np.ndarray) -> FitResult:
    x, y = sort_xy(x, y)
    local_max = float(np.max(y))
    if local_max <= 0:
        raise ValueError("Intensity maximum must be positive for bounded peak fitting.")

    p0 = [
        local_max * 0.2,
        780.5,
        1.0,
        local_max * 0.6,
        780.0,
        1.2,
        local_max * 0.4,
        781.5,
        1.2,
    ]
    bounds = (
        [0, 775, 0.1, 0, 779.5, 0.5, 0, 781.0, 0.5],
        [local_max, 785, 3.0, np.inf, 780.5, 2.5, np.inf, 782.5, 2.5],
    )

    model = make_model(y)
    popt, pcov = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=20000)
    bg_a, bg_b, bg_c, a1, c1, w1, a2, c2, w2 = popt

    background = adaptive_background(x, y, c1, c2, bg_a, bg_b, bg_c)
    co2 = gaussian(x, a1, c1, w1)
    co3 = gaussian(x, a2, c2, w2)
    fitted = background + co2 + co3

    area1 = float(np.trapz(co2, x))
    area2 = float(np.trapz(co3, x))
    total_area = area1 + area2
    if total_area <= 0:
        raise ValueError("Peak areas are not positive; check initial guesses and bounds.")

    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    return FitResult(
        params=popt,
        covariance=pcov,
        x=x,
        y=y,
        background=background,
        co2=co2,
        co3=co3,
        fitted=fitted,
        co2_ratio=area1 / total_area,
        co3_ratio=area2 / total_area,
        r2=r2,
    )


def plot_fit(result: FitResult, output: Path | None = None, show: bool = False) -> None:
    _, _, _, _, c1, _, _, c2, _ = result.params
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    ax.plot(result.x, result.y, "o", ms=3.5, color="#F4A261", alpha=0.6, label="Raw Data")
    ax.plot(result.x, result.fitted, "-", lw=2, color="black", label=f"Fit ($R^2$={result.r2:.3f})")
    ax.plot(
        result.x,
        result.co2 + result.background,
        "--",
        lw=2,
        color="#4C72B0",
        label=f"$Co^{{+2}}$ ({c1:.2f} eV)",
    )
    ax.plot(
        result.x,
        result.co3 + result.background,
        "--",
        lw=2,
        color="#55A868",
        label=f"$Co^{{+3}}$ ({c2:.2f} eV)",
    )

    mask = (result.x >= 768) & (result.x <= 785)
    ax.fill_between(
        result.x[mask],
        result.background[mask],
        (result.co2 + result.background)[mask],
        color="#4C72B0",
        alpha=0.4,
    )
    ax.fill_between(
        result.x[mask],
        result.background[mask],
        (result.co3 + result.background)[mask],
        color="#55A868",
        alpha=0.4,
    )

    ax.set_xlim(float(np.max(result.x)), 769)
    ax.set_ylim(-50, float(np.max(result.y)) * 1.15)
    ax.set_title("Co 2p", fontsize=20)
    ax.set_box_aspect(1)
    ax.set_xlabel("Binding Energy (eV)", fontsize=18)
    ax.set_ylabel("Intensity (a.u.)", fontsize=18)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.text(
        0.98,
        0.92,
        f"$Co^{{+2}}$ = {result.co2_ratio:.2f}\n$Co^{{+3}}$ = {result.co3_ratio:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=14,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_components(result: FitResult, output: Path) -> None:
    arr = np.column_stack(
        [
            result.x,
            result.y,
            result.background,
            result.co2,
            result.co3,
            result.fitted,
        ]
    )
    np.savetxt(
        output,
        arr,
        delimiter=",",
        header="binding_energy_eV,intensity,background,co2_peak,co3_peak,fitted",
        comments="",
    )

# ========= 数据 =========
data = np.array([
[788.895,68.8616],[788.795,71.3104],[788.695,74.1288],[788.595,79.4152],
[788.495,85.2177],[788.395,91.2016],[788.295,96.5239],[788.195,100.945],
[788.095,104.323],[787.995,106.709],[787.895,108.173],[787.795,108.713],
[787.695,108.22],[787.595,106.561],[787.495,103.662],[787.395,99.5645],
[787.295,94.4949],[787.195,88.8014],[787.095,82.8828],[786.995,77.1107],
[786.895,71.7684],[786.795,67.0181],[786.695,62.9488],[786.595,59.5957],
[786.495,56.9476],[786.395,54.9588],[786.295,53.5341],[786.195,52.5065],
[786.095,51.6568],[785.995,50.7569],[785.895,49.5974],[785.795,48.0197],
[785.695,45.926],[785.595,43.2672],[785.495,40.0313],[785.395,36.2354],
[785.295,31.9522],[785.195,27.321],[785.095,22.559],[784.995,17.9235],
[784.895,13.6887],[784.795,10.0702],[784.695,7.19845],[784.595,5.0835],
[784.495,3.67347],[784.395,2.87842],[784.295,2.61186],[784.195,2.80432],
[784.095,3.44219],[783.995,4.50849],[783.895,5.97888],[783.795,7.84877],
[783.695,10.1573],[783.595,12.9788],[783.495,16.47],[783.395,20.8505],
[783.295,26.3423],[783.195,33.1028],[783.095,41.198],[782.995,50.5707],
[782.895,61.0709],[782.795,72.5088],[782.695,84.7463],[782.595,97.7315],
[782.495,111.533],[782.395,126.315],[782.295,142.317],[782.195,159.771],
[782.095,178.902],[781.995,199.9],[781.895,222.935],[781.795,248.168],
[781.695,275.804],[781.595,306.078],[781.495,339.26],[781.395,375.622],
[781.295,415.403],[781.195,458.727],[781.095,505.548],[780.995,555.602],
[780.895,608.416],[780.795,663.318],[780.695,719.527],[780.595,776.258],
[780.495,832.799],[780.395,888.524],[780.295,942.901],[780.195,995.428],
[780.095,1045.53],[779.995,1092.47],[779.895,1135.38],[779.795,1173.29],
[779.695,1205.13],[779.595,1229.87],[779.495,1246.55],[779.395,1254.24],
[779.295,1252.11],[779.195,1239.35],[779.095,1215.35],[778.995,1179.73],
[778.895,1132.61],[778.795,1074.76],[778.695,1007.67],[778.595,933.432],
[778.495,854.627],[778.395,773.955],[778.295,693.948],[778.195,616.744],
[778.095,544.004],[777.995,476.881],[777.895,416.104],[777.795,362.084],
[777.695,314.969],[777.595,274.634],[777.495,240.692],[777.395,212.507],
[777.295,189.213],[777.195,169.792],[777.095,153.225],[776.995,138.627],
[776.895,125.33],[776.795,112.949],[776.695,101.421],[776.595,90.913],
[776.495,81.7002],[776.395,74.0763],[776.295,68.2305],[776.195,64.1408],
[776.095,61.5609],[775.995,60.0968],[775.895,59.2521],[775.795,58.541],
[775.695,57.583],[775.595,56.1604],[775.495,54.1897],[775.395,51.7373],
[775.295,48.9557],[775.195,46.0483],[775.095,43.2106],[774.995,40.6218],
[774.895,38.3637],[774.795,36.4147],[774.695,34.6338],[774.595,32.8025],
[774.495,30.6486],[774.395,27.9663],[774.295,24.6764],[774.195,20.8656],
[774.095,16.7653],[773.995,12.7097],[773.895,9.02295],[773.795,5.93882],
[773.695,3.54135],[773.595,1.76232],[773.495,0.434224],[773.395,-0.643693],
[773.295,-1.66073],[773.195,-2.73097],[773.095,-3.88482],[772.995,-5.11838],
[772.895,-6.40973],[772.795,-7.7091],[772.695,-8.98177],[772.595,-10.1917],
[772.495,-11.2665],[772.395,-12.0936],[772.295,-12.5507],[772.195,-12.49],
[772.095,-11.775],[771.995,-10.3042],[771.895,-8.03231],[771.795,-4.96065],
[771.695,-1.14804],[771.595,3.29074],[771.495,8.17421],[771.395,13.2605],
[771.295,18.2381],[771.195,22.7401],[771.095,26.3988],[770.995,28.9031],
[770.895,30.0587],[770.795,29.8341],[770.695,28.3599],[770.595,25.8735],
[770.495,22.6586],[770.395,18.9463],[770.295,14.8673],[770.195,10.4684],
[770.095,5.74367],[769.995,0.680027],[769.895,-4.65408],[769.795,-10.1009],
[769.695,-15.4683],[769.595,-20.5456],[769.495,-25.1467],[769.395,-29.1498],
[769.295,-32.5238],[769.195,-35.1646],[769.095,-37.1628],[768.995,-38.1552],
[768.895,-38.9231]])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Co 2p XPS data with two Gaussian peaks and an adaptive background.")
    parser.add_argument("--output", default="co_2p_fit.png", help="Path for the fitted figure.")
    parser.add_argument("--components", default="co_2p_components.csv", help="CSV path for fitted components.")
    parser.add_argument("--show", action="store_true", help="Show the plot window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plot()

    x = data[:, 0]
    y = data[:, 1]
    result = fit_co_2p(x, y)

    output = Path(args.output) if args.output else None
    components = Path(args.components) if args.components else None
    plot_fit(result, output=output, show=args.show)
    if components is not None:
        write_components(result, components)

    _, _, _, _, c1, w1, _, c2, w2 = result.params
    print(f"Co2+ center = {c1:.4f} eV, width = {w1:.4f} eV, ratio = {result.co2_ratio:.4f}")
    print(f"Co3+ center = {c2:.4f} eV, width = {w2:.4f} eV, ratio = {result.co3_ratio:.4f}")
    print(f"R2 = {result.r2:.5f}")
    if output is not None:
        print(f"Saved figure: {output}")
    if components is not None:
        print(f"Saved components: {components}")


if __name__ == "__main__":
    main()
