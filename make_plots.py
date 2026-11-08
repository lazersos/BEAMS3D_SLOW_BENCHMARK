#!/usr/bin/env python3
"""BEAMS3D benchmark plots converted from make_plots.m.

Requirements: STELLOPT pySTEL (libstell), numpy, scipy, matplotlib, h5py,
and the LIBSTELL shared library needed by pySTEL's VMEC reader.
Install pySTEL using STELLOPT's build instructions, or put STELLOPT/pySTEL
on PYTHONPATH and configure STELLOPT_PATH for your LIBSTELL installation.

Run from the directory containing beams3d_ORBITS*.h5 and the corresponding
wout_ORBITS*.nc files:
    python make_plots.py
    python make_plots.py --data-dir results --old-dir previous --output-dir plots
    python make_plots.py --old-dir ../old --second-old-dir ../older

The old data directory defaults to the parent of data-dir. --second-old-dir
adds an independent second previous run to all four benchmark plots.
--no-old disables only the first old run. Both previous runs match
current filenames and use the current run's VMEC equilibria, as in MATLAB.
VENUS-LEVIS data are read from data-dir/VL_results.txt if present;
--venus-file overrides this.
Only BEAMS3D HDF5 files are read. Output is four PNGs plus vector PDFs.

Theory references assume a uniform, single-ion-species Maxwellian plasma.
Defaults: ne=1.2e19 m^-3, Te=1000 eV, Ti=100 eV, ln(Lambda)=15.84,
and singly charged beam/background ions. The background-ion mass comes from
plasma_mass in the current BEAMS3D file. Set --theory-* options to match the
simulation. For ion charge Zi, quasineutrality sets ni=ne/Zi.
The additional Maxwellian curves use COLLISIONS.collisionfreq_energy, with
P_species = E * nu_energy in W per test particle. Negative P is energy gained
by the test particle from that background species; it is not clipped.
Both theory fractions are instantaneous ratios, not lifetime-integrated
deposition fractions. The energy curves use the initial-rate approximation
Delta E = (Pe+Pi) * theory_time; they do not solve the Fokker-Planck equation
or follow the changing particle energy during that interval.
For the heating plot, --theory-power-scale multiplies both per-particle power
references. Its default 0.001 preserves the MATLAB benchmark normalization;
it represents an effective particle population, not a units conversion.
Set it to 1 for per-particle W, or to the population matching marker weights.

Conversion choices:
* Retain the original slow026 exclusion, but do not delete an arbitrary 15th
  point. Use --exclude TOKEN (repeatable) for additional filename exclusions.
* Keep the MATLAB interpolation coordinates: rho for heating fraction, s for
  total power. Total power uses sum(P * dV/drho) * drho. The fraction's common
  1/drho factor cancels, so that factor is omitted.
* Energy loss is the unweighted marker mean between stored rows 1 and 1000
  (Python indices 0 and 999), without time rescaling. --energy-row uses MATLAB
  one-based indexing. Too-short files yield no energy point and a warning.
* Default theory parameters retain the original Wesson benchmark constants
  and 1 ms interval. Both models consistently use plasma_mass for background
  ions, including the energy plot and the unequal-mass heating fraction.
* Missing or invalid measurements remain NaN, never artificial zero points.
  Heating, pitch, and energy failures are reported independently.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ME = 9.10938356e-31
EC = 1.60217662e-19
EPS0 = 8.8541878128e-12


@dataclass
class Measurement:
    energy: float
    filename: str
    fraction: float = np.nan
    qe: float = np.nan
    qi: float = np.nan
    pitch_span: float = np.nan
    energy_loss: float = np.nan
    mass: float = np.nan
    plasma_mass: float = np.nan


@dataclass(frozen=True)
class TheoryParameters:
    electron_density: float = 1.2e19
    electron_temperature_ev: float = 1000.0
    ion_temperature_ev: float = 100.0
    coulomb_log: float = 15.84
    beam_charge_number: float = 1.0
    ion_charge_number: float = 1.0
    interval_s: float = 1e-3
    power_scale: float = 1e-3


def warn(path: Path, quantity: str, error: Exception) -> None:
    warnings.warn(f"{path}: {quantity}: {error}", stacklevel=2)


def summed_profile(profile: np.ndarray, ns: int) -> np.ndarray:
    """Sum beams for either (radial bin, beam) or (beam, radial bin)."""
    array = np.asarray(profile, dtype=float)
    if array.ndim not in (1, 2):
        raise ValueError(f"expected a 1D or 2D power profile, got {array.shape}")
    radial_axes = [axis for axis, length in enumerate(array.shape) if length == ns]
    if not radial_axes:
        raise ValueError(f"expected an axis with {ns} radial bins, got {array.shape}")
    if len(radial_axes) > 1 and ns > 1:
        raise ValueError(f"ambiguous radial axis for power profile {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite power profile")
    array = np.moveaxis(array, radial_axes[0], 0)
    return array.reshape(ns, -1).sum(axis=1)


def heating(beam, vmec) -> tuple[float, float, float]:
    ns = int(beam.ns_prof1)
    vp = np.asarray(vmec.vp, dtype=float).reshape(-1)
    if ns < 1 or vp.size != int(vmec.ns) or vp.size < 2:
        raise ValueError("inconsistent radial grid")
    s = np.linspace(0.0, 1.0, vp.size)
    rho = np.sqrt(s)
    dvdrho = 2.0 * rho * vp * (4.0 * np.pi**2)
    rho_beam = (np.arange(ns) + 0.5) / ns
    pe = summed_profile(beam.epower_prof, ns)
    pi = summed_profile(beam.ipower_prof, ns)
    weight_fraction = PchipInterpolator(rho, dvdrho)(rho_beam)
    qe_fraction = float(pe @ weight_fraction)
    qi_fraction = float(pi @ weight_fraction)
    total = qe_fraction + qi_fraction
    fraction = qe_fraction / total if total != 0.0 else np.nan
    weight_power = PchipInterpolator(s, dvdrho)(rho_beam**2) / ns
    return fraction, float(pe @ weight_power), float(pi @ weight_power)


def orbit_quantities(beam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vll = np.asarray(beam.vll_lines, dtype=float)
    vperp = np.asarray(beam.calcVperp(), dtype=float)
    bfield = np.asarray(beam.B_lines, dtype=float)
    mass = np.asarray(beam.mass, dtype=float).reshape(-1)
    if vll.ndim != 2 or vll.shape != vperp.shape or bfield.shape != vll.shape:
        raise ValueError("expected matching (stored row, particle) orbit arrays")
    if mass.size != vll.shape[1] or np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("invalid marker masses")
    speed2 = vll**2 + vperp**2
    # pySTEL uses B < 0 as an invalid stored-point sentinel.
    valid = (bfield >= 0) & np.isfinite(speed2) & (speed2 >= 0)
    return speed2, valid, mass


def read_measurement(path: Path, energy: float, vmec_path: Path,
                     energy_row: int) -> Measurement:
    from libstell.beams3d import BEAMS3D
    from libstell.vmec import VMEC

    result = Measurement(energy, path.name)
    beam = BEAMS3D()
    try:
        beam.read_beams3d(str(path))
    except Exception as error:
        warn(path, "BEAMS3D read failed", error)
        return result
    try:
        result.mass = float(np.asarray(beam.mass).reshape(-1)[0])
        result.plasma_mass = float(beam.plasma_mass)
    except (AttributeError, ValueError, IndexError) as error:
        warn(path, "theory masses unavailable", error)
    try:
        vmec = VMEC()
        vmec.read_wout(str(vmec_path))
        result.fraction, result.qe, result.qi = heating(beam, vmec)
    except Exception as error:
        warn(path, "heating unavailable", error)
    try:
        speed2, valid, mass = orbit_quantities(beam)
        alpha = np.full(speed2.shape, np.nan)
        np.divide(beam.vll_lines, np.sqrt(speed2), out=alpha, where=valid & (speed2 > 0))
        if np.any(np.isfinite(alpha)):
            result.pitch_span = float(np.nanmax(alpha) - np.nanmin(alpha))
        row = energy_row - 1
        if row >= speed2.shape[0]:
            raise ValueError(f"requested row {energy_row}, only {speed2.shape[0]} stored")
        keep = valid[0] & valid[row]
        if not np.any(keep):
            raise ValueError("no markers valid at both energy endpoints")
        delta = 0.5 * (speed2[0, keep] - speed2[row, keep]) * mass[keep] / EC
        result.energy_loss = float(np.mean(delta))
    except Exception as error:
        warn(path, "orbit/energy calculation incomplete", error)
    return result


def read_venus(path: Path) -> np.ndarray | None:
    """Read comma- or whitespace-separated numeric rows after a text preamble.

    Columns: particle energy [keV], electron energy [eV], ion energy [eV],
    minimum pitch, maximum pitch. Preserve signed deposited energies.
    """
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        try:
            values = [float(word.replace("D", "E").replace("d", "e"))
                      for word in re.split(r"[,\s]+", line.strip()) if word]
        except ValueError:
            continue
        if len(values) >= 5 and np.all(np.isfinite(values[:5])):
            rows.append(values[:5])
    if not rows:
        raise ValueError(f"no numeric rows with five columns in {path}")
    data = np.asarray(rows)
    return data[np.argsort(data[:, 0])]


def theory(x: np.ndarray, marker_mass: float, ion_mass: float,
           params: TheoryParameters = TheoryParameters()):
    """Asymptotic positive energy-loss rates, W per test particle."""
    energy_j = np.asarray(x, dtype=float) * params.electron_temperature_ev * EC
    te_j = params.electron_temperature_ev * EC
    ad = (params.electron_density * params.beam_charge_number**2 * EC**4
          * params.coulomb_log / (2.0 * np.pi * EPS0**2 * marker_mass**2))
    qe = (2.0 * np.sqrt(ME) * marker_mass * ad * energy_j
          / (3.0 * np.sqrt(2.0 * np.pi) * te_j**1.5))
    qi = (marker_mass**2.5 * ad * params.ion_charge_number
          / (2.0**1.5 * ion_mass * np.sqrt(energy_j)))
    return qe, qi


def maxwellian_theory(x: np.ndarray, marker_mass: float, ion_mass: float,
                      params: TheoryParameters = TheoryParameters()):
    """Net mean energy-loss rates using pySTEL's Maxwellian energy operator.

    collisionfreq_energy is defined by d<E>/dt = -nu_energy * E, so no
    extra factor of two is needed. Its temperatures are in eV, density in
    m^-3, masses in kg, and charges are charge numbers (not coulombs).
    """
    from libstell.collisions import COLLISIONS

    collisions = COLLISIONS()
    energy_j = np.asarray(x, dtype=float) * params.electron_temperature_ev * EC
    speed = np.sqrt(2.0 * energy_j / marker_mass)
    nu_e = collisions.collisionfreq_energy(
        marker_mass, params.beam_charge_number, speed, ME, -1.0,
        params.electron_temperature_ev, params.electron_density, params.coulomb_log)
    nu_i = collisions.collisionfreq_energy(
        marker_mass, params.beam_charge_number, speed, ion_mass,
        params.ion_charge_number, params.ion_temperature_ev,
        params.electron_density / params.ion_charge_number, params.coulomb_log)
    return energy_j * nu_e, energy_j * nu_i


def electron_fraction(pe, pi):
    """Signed instantaneous fraction where the net particle energy loss is positive."""
    pe, pi = np.broadcast_arrays(np.asarray(pe, dtype=float), np.asarray(pi, dtype=float))
    total = pe + pi
    fraction = np.full(total.shape, np.nan)
    np.divide(pe, total, out=fraction,
              where=np.isfinite(pe) & np.isfinite(pi) & (total > 0))
    return fraction


def make_figures(current, old, venus, output: Path, second_old=(),
                 theory_params: TheoryParameters = TheoryParameters()) -> None:
    plt.rcParams.update({"font.size": 24, "lines.linewidth": 4,
                         "lines.markersize": 18, "lines.markeredgewidth": 3,
                         "figure.facecolor": "white"})
    energy = np.array([item.energy for item in current])
    def points(ax, records, field, fmt, label):
        x = np.array([item.energy for item in records])
        y = np.array([getattr(item, field) for item in records])
        keep = np.isfinite(x) & np.isfinite(y)
        if np.any(keep):
            ax.plot(x[keep], y[keep], fmt, label=label, markerfacecolor="none")

    def figure(title, ylabel, xmax=500):
        fig, ax = plt.subplots(figsize=(10.24, 7.68), dpi=100)
        ax.set(xscale="log", xlim=(1, xmax), xlabel=r"$E_{\mathrm{Beam}}/T_e$",
               ylabel=ylabel, title=title)
        ax.set_xticks([1, 10, 100], labels=["1", "10", "100"])
        return fig, ax

    def save(fig, ax, basename, location):
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            many_entries = len(handles) > 8
            ax.legend(loc=location, fontsize=14 if many_entries else 17,
                      ncol=2 if many_entries else 1)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{basename}.{suffix}")
        plt.close(fig)

    masses = [item for item in current if np.isfinite(item.mass) and item.mass > 0]
    reference = masses[0] if masses else None
    if reference and not np.allclose([item.mass / reference.mass for item in masses], 1):
        warnings.warn("marker species differ across files; theory uses the first file's mass")
    x_fraction = np.arange(1.0, 500.005, 0.01)
    x_theory = np.arange(1.0, max(1.0, np.floor(np.max(energy))) + 1.0)
    theory_reference = (reference is not None and np.isfinite(reference.plasma_mass)
                        and reference.plasma_mass > 0)
    if reference and not theory_reference:
        warnings.warn("background-ion mass unavailable; theory curves omitted")
    if theory_reference:
        asym_fraction = theory(x_fraction, reference.mass, reference.plasma_mass,
                               theory_params)
        maxwell_fraction = maxwellian_theory(x_fraction, reference.mass,
                                            reference.plasma_mass, theory_params)
        qe, qi = theory(x_theory, reference.mass, reference.plasma_mass, theory_params)
        qe_maxwell, qi_maxwell = maxwellian_theory(
            x_theory, reference.mass, reference.plasma_mass, theory_params)

    fig, ax = figure("Fractional Heating Benchmark", "Electron Heating Fraction")
    points(ax, current, "fraction", "ok", "BEAMS3D")
    points(ax, old, "fraction", "xb", "BEAMS3D (old)")
    points(ax, second_old, "fraction", "+m", "BEAMS3D (old 2)")
    if theory_reference:
        ax.plot(x_fraction, electron_fraction(*asym_fraction),
                "r", label="Theory (asymptotic)")
        ax.plot(x_fraction, electron_fraction(*maxwell_fraction),
                color="darkorange", linestyle="-.", label="COLLISIONS (Maxwellian)")
    ax.set_yticks([0, 0.5, 1.0], labels=["0", "0.5", "1.0"])
    save(fig, ax, "benchmark_a10_slow", "upper left")

    fig, ax = figure("Pitch Angle Scattering Benchmark", r"$\Delta(v_\parallel/v)$")
    points(ax, current, "pitch_span", "ok", "BEAMS3D")
    points(ax, old, "pitch_span", "xb", "BEAMS3D (old)")
    points(ax, second_old, "pitch_span", "+m", "BEAMS3D (old 2)")
    ax.set_yticks([0, 1, 2], labels=["0", "1.0", "2.0"])
    save(fig, ax, "benchmark_a10_pitch", "upper right")

    fig, ax = figure("Total Heating", "Heating [W]", 1000)
    points(ax, current, "qe", "ok", r"$Q_e^{\mathrm{BEAMS3D}}$")
    points(ax, current, "qi", "sk", r"$Q_i^{\mathrm{BEAMS3D}}$")
    points(ax, old, "qe", "xb", r"$Q_e$ BEAMS3D (old)")
    points(ax, old, "qi", "sb", r"$Q_i$ BEAMS3D (old)")
    points(ax, second_old, "qe", "+m", r"$Q_e$ BEAMS3D (old 2)")
    points(ax, second_old, "qi", "Dm", r"$Q_i$ BEAMS3D (old 2)")
    if venus is not None:
        ax.plot(venus[:, 0], venus[:, 1] * EC / 50.0, "^g", label="VENUS-LEVIS electrons")
        ax.plot(venus[:, 0], venus[:, 2] * EC / 50.0, "vg", label="VENUS-LEVIS ions")
    if theory_reference:
        scale = theory_params.power_scale
        ax.plot(x_theory, qe * scale, "r", label=r"$Q_e$ asymptotic")
        ax.plot(x_theory, qi * scale, "r--", label=r"$Q_i$ asymptotic")
        ax.plot(x_theory, qe_maxwell * scale, color="darkorange",
                label=r"$Q_e$ COLLISIONS")
        ax.plot(x_theory, qi_maxwell * scale, color="darkorange", linestyle="--",
                label=r"$Q_i$ COLLISIONS")
    save(fig, ax, "benchmark_a10_slow_power", "upper left")

    fig, ax = figure("Particle Energy Change", r"$\Delta E$ [eV]", 1000)
    points(ax, current, "energy_loss", "ok", "BEAMS3D")
    points(ax, old, "energy_loss", "xb", "BEAMS3D (old)")
    points(ax, second_old, "energy_loss", "+m", "BEAMS3D (old 2)")
    if venus is not None:
        ax.plot(venus[:, 0], (venus[:, 1] + venus[:, 2]) / 50.0, "^g", label="VENUS-LEVIS")
    if theory_reference:
        ax.plot(x_theory, (qe + qi) * theory_params.interval_s / EC,
                "r", label="Asymptotic (initial rate)")
        ax.plot(x_theory, (qe_maxwell + qi_maxwell) * theory_params.interval_s / EC,
                color="darkorange", linestyle="-.", label="COLLISIONS (initial rate)")
    save(fig, ax, "benchmark_a10_slow_energy", "lower right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--old-dir", type=Path, help="default: parent of data-dir")
    parser.add_argument("--no-old", action="store_true", help="disable the first old run")
    parser.add_argument("--second-old-dir", type=Path,
                        help="optional second old run for all four benchmark plots")
    parser.add_argument("--venus-file", type=Path, help="default: data-dir/VL_results.txt")
    parser.add_argument("--no-venus", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--energy-row", type=int, default=1000, help="one-based stored row")
    parser.add_argument("--exclude", action="append", default=[], metavar="TOKEN",
                        help="exclude filenames containing TOKEN, in addition to slow026")
    defaults = TheoryParameters()
    group = parser.add_argument_group("uniform-plasma theory parameters")
    group.add_argument("--theory-ne", type=float, default=defaults.electron_density,
                       help="electron density in m^-3 (default: 1.2e19)")
    group.add_argument("--theory-te", type=float, default=defaults.electron_temperature_ev,
                       help="electron temperature in eV (default: 1000)")
    group.add_argument("--theory-ti", type=float, default=defaults.ion_temperature_ev,
                       help="ion temperature in eV (default: 100)")
    group.add_argument("--theory-coulomb-log", type=float, default=defaults.coulomb_log,
                       help="common Coulomb logarithm (default: 15.84)")
    group.add_argument("--theory-beam-charge", type=float, default=defaults.beam_charge_number,
                       help="beam charge number (default: 1)")
    group.add_argument("--theory-ion-charge", type=float, default=defaults.ion_charge_number,
                       help="background ion charge number; ni=ne/Zi (default: 1)")
    group.add_argument("--theory-time", type=float, default=defaults.interval_s,
                       help="initial-rate energy-loss interval in seconds (default: 0.001)")
    group.add_argument("--theory-power-scale", type=float, default=defaults.power_scale,
                       help="effective population scaling per-particle W (default: 0.001)")
    args = parser.parse_args()
    theory_params = TheoryParameters(
        args.theory_ne, args.theory_te, args.theory_ti, args.theory_coulomb_log,
        args.theory_beam_charge, args.theory_ion_charge, args.theory_time,
        args.theory_power_scale)
    for name, value in vars(theory_params).items():
        if not np.isfinite(value) or value <= 0:
            parser.error(f"theory parameter {name} must be finite and positive")
    if args.energy_row < 2:
        parser.error("--energy-row must be at least 2")
    if args.second_old_dir is not None and not args.second_old_dir.is_dir():
        parser.error(f"second old run directory does not exist: {args.second_old_dir}")
    files = []
    for path in sorted(args.data_dir.glob("beams3d_ORBITS*.h5")):
        if any(token in path.name for token in ["slow026", *args.exclude]):
            continue
        match = re.search(r"^beams3d_ORBITS\D*(\d+)", path.stem)
        if match is None or int(match.group(1)) <= 0:
            warnings.warn(f"{path}: cannot extract positive E_Beam/T_e from filename")
            continue
        files.append((int(match.group(1)), path))
    if not files:
        parser.error(f"no eligible beams3d_ORBITS*.h5 files in {args.data_dir}")
    files.sort(key=lambda pair: (pair[0], pair[1].name))
    try:
        from libstell.beams3d import BEAMS3D  # noqa: F401
        from libstell.vmec import VMEC  # noqa: F401
        from libstell.collisions import COLLISIONS  # noqa: F401
    except ImportError as error:
        parser.error(f"pySTEL is unavailable: {error}. See requirements above.")
    old_dir = args.old_dir if args.old_dir is not None else args.data_dir.resolve().parent
    current, old, second_old = [], [], []
    for energy, path in files:
        vmec_path = args.data_dir / f"wout_{path.stem.removeprefix('beams3d_')}.nc"
        current.append(read_measurement(path, energy, vmec_path, args.energy_row))
        old_path = old_dir / path.name
        if not args.no_old and old_path.is_file():
            old.append(read_measurement(old_path, energy, vmec_path, args.energy_row))
        if args.second_old_dir is not None:
            second_old_path = args.second_old_dir / path.name
            if second_old_path.is_file():
                second_old.append(read_measurement(second_old_path, energy, vmec_path,
                                                   args.energy_row))
    if not args.no_old and not old:
        warnings.warn(f"no matching old BEAMS3D files in {old_dir}")
    if args.second_old_dir is not None and not second_old:
        warnings.warn(f"no matching second old BEAMS3D files in {args.second_old_dir}")
    venus = None
    if not args.no_venus:
        venus_path = args.venus_file or args.data_dir / "VL_results.txt"
        if args.venus_file is not None and not venus_path.is_file():
            parser.error(f"VENUS-LEVIS file does not exist: {venus_path}")
        venus = read_venus(venus_path)
    if not any(np.isfinite(getattr(item, field)) for item in current
               for field in ("fraction", "qe", "qi", "pitch_span", "energy_loss")):
        parser.error("no valid BEAMS3D measurements; see warnings above")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_figures(current, old, venus, args.output_dir, second_old=second_old,
                 theory_params=theory_params)
    print(f"Saved four benchmark plots as PNG and PDF in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
