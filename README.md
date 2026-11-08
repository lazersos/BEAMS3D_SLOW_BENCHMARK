# BEAMS3D Slowing-Down Benchmark

Benchmark inputs and analysis tools for studying collisional slowing down,
electron and ion heating, and pitch-angle scattering with
[BEAMS3D](https://github.com/PrincetonUniversity/STELLOPT/tree/master/BEAMS3D).
The cases follow initially monoenergetic ions in a prescribed, axisymmetric
VMEC equilibrium and compare BEAMS3D results with analytic and Maxwellian
collision-model references. Previous BEAMS3D runs and optional VENUS-LEVIS
results can also be overlaid.

This repository supplies the input files, VMEC equilibria, run script, and
plotting script. BEAMS3D and its dependencies must be installed separately;
simulation results are generated locally.

## Repository contents

| File | Purpose |
| --- | --- |
| `input.ORBITS_slowNNN` | Combined VMEC `INDATA` and `BEAMS3D_INPUT` namelists for each case. |
| `wout_ORBITS_slowNNN.nc` | Corresponding precomputed VMEC equilibria. |
| [`run.sh`](run.sh) | Runs all 24 cases sequentially through MPI. |
| [`make_plots.py`](make_plots.py) | Reads BEAMS3D HDF5 output and generates four benchmark figures in PNG and PDF formats. |
| [`LICENSE`](LICENSE) | MIT license. |

## Benchmark configuration

The main scan contains 23 deuterium cases at initial energies of
1–10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, and 500 keV.
For these cases, the filename suffix denotes the initial energy in keV.

| Parameter | Supplied configuration |
| --- | --- |
| Equilibrium geometry | Axisymmetric, circular boundary; major radius 10 m and minor radius 1 m. |
| VMEC pressure | Zero (`PRES_SCALE = 0`); kinetic profiles are specified separately for BEAMS3D. |
| Test and background ions | Singly charged deuterium, mass approximately `3.3445e-27` kg. |
| Markers per case | 250, with identical initial position and velocity parameters. |
| Initial position | `R = 10.25 m`, `Z = 0`, `phi = 0`. |
| Initial pitch angle | Approximately 5 degrees, as specified in the input setup comments. |
| Follow time | 1 ms. |
| Orbit output setting | `NPOINC = 1000`. |
| Electron density | `1.2e19 m^-3` at profile knots `s = 0, 0.2, 0.4, 0.6, 0.8`; zero at `s = 1`. |
| Electron and ion temperatures | Both 1000 eV at the same interior knots; zero at `s = 1`. |
| BEAMS3D grid | `128 × 128 × 32` in R, Z, and phi. |
| Orbit integrator | `LSODE`, with `FOLLOW_TOL = 1e-8`. |

`slow026` is an additional hydrogen test near 25 keV, with hydrogen marker
and background-ion masses. Its suffix is a case identifier rather than its
energy. The run script includes it, but the plotting script always excludes
filenames containing `slow026`.

Some input files retain copied energy values in their setup comments. Use
the active `MASS_IN`, `VLL_START_IN`, and `MU_START_IN` values when modifying
a case.

## Requirements

To run the simulations, install an MPI-enabled `xbeams3d` executable from
[STELLOPT](https://github.com/PrincetonUniversity/STELLOPT), configured to
produce HDF5 output, and make it available on `PATH`. Use an MPI launcher
compatible with that build. The supplied `run.sh` uses Zsh and the
Open MPI option `--use-hwthread-cpu`.

For analysis, use Python 3.10 or newer, NumPy, SciPy, Matplotlib, h5py, and
STELLOPT's pySTEL package (`libstell`). The pySTEL VMEC reader also requires
the compiled LIBSTELL shared library. Follow the
[STELLOPT build and installation guidance](https://github.com/PrincetonUniversity/STELLOPT)
for the simulation executable and shared library.

For an existing STELLOPT installation, a typical analysis setup is:

```bash
python3 -m pip install numpy scipy matplotlib h5py

# Replace this path with your STELLOPT installation.
export STELLOPT_PATH=/path/to/STELLOPT
export PYTHONPATH="$STELLOPT_PATH/pySTEL${PYTHONPATH:+:$PYTHONPATH}"
```

Use a pySTEL version that provides `libstell.collisions.COLLISIONS` and
its `collisionfreq_energy` method for the Maxwellian reference curves.

## Run the benchmark

Clone the repository and run commands from its root so that BEAMS3D can
find the matching input and equilibrium files:

```bash
git clone https://github.com/lazersos/BEAMS3D_SLOW_BENCHMARK.git
cd BEAMS3D_SLOW_BENCHMARK

# Run one case first; choose a rank count suitable for your allocation.
mpiexec -n 4 xbeams3d -vmec ORBITS_slow001 -plasma -collisions

# Run the full scan using the supplied Open MPI commands.
zsh run.sh
```

`run.sh` does not set a rank count explicitly. Adjust its launcher options
for your machine or scheduler before running the full scan. With another
MPI implementation, replace `--use-hwthread-cpu` with supported options.

Each successful case should produce an HDF5 file such as
`beams3d_ORBITS_slow001.h5`. Retain the corresponding
`wout_ORBITS_slow001.nc` alongside it for analysis. The supplied equilibria
allow the benchmark to be run without first regenerating VMEC solutions.

## Generate plots

After running the simulations:

```bash
python3 make_plots.py --no-old --theory-ti 1000 --output-dir plots
```

The explicit `--theory-ti 1000` matches the interior ion temperature in
the supplied inputs. The script's historical default is **100 eV**.
`--no-old` disables the automatic search for a previous run in the parent
directory. Plotting uses Matplotlib's `Agg` backend and requires no display.

| Output basename | Quantity |
| --- | --- |
| `benchmark_a10_slow` | Volume-weighted electron heating fraction, `Qe / (Qe + Qi)`. |
| `benchmark_a10_pitch` | Pitch span, `max(v_parallel / v) - min(v_parallel / v)`, across valid stored points and markers. |
| `benchmark_a10_slow_power` | Volume-integrated electron and ion heating powers. |
| `benchmark_a10_slow_energy` | Unweighted mean marker energy loss between stored rows 1 and 1000. |

Each figure is saved as both `.png` and `.pdf`, for eight files in total.
The horizontal coordinate is extracted from the numeric filename suffix
and labeled `E_Beam / Te`; it agrees with the supplied deuterium energy scan
because `Te = 1 keV`. Changing `--theory-te` does not rescale the measured
x coordinates, so adapt the analysis if changing the scan's normalization.

### Compare previous runs

```bash
python3 make_plots.py \
  --data-dir . \
  --old-dir ../previous-run \
  --second-old-dir ../older-run \
  --theory-ti 1000 \
  --output-dir plots
```

Previous runs are matched to current HDF5 filenames and analyzed using the
**current run's VMEC equilibria**. If `--old-dir` is omitted, it defaults to
the parent of `--data-dir`. `--no-old` disables only the first previous run;
an explicitly supplied `--second-old-dir` remains active. When using a
separate data directory, put both current HDF5 files and matching VMEC
files there.

### Optional VENUS-LEVIS data

Place `VL_results.txt` in the data directory or pass `--venus-file PATH`.
Numeric rows may be comma- or whitespace-separated, following an optional
text preamble. The first five columns are:

| Column | Value |
| --- | --- |
| 1 | Particle energy in keV. |
| 2 | Electron energy in eV. |
| 3 | Ion energy in eV. |
| 4 | Minimum pitch. |
| 5 | Maximum pitch. |

The script overlays VENUS-LEVIS data on the total-heating and energy-change
figures. Its existing normalization divides electron and ion energies by
50, and additionally multiplies by the elementary charge for the heating
plot. Confirm that this normalization matches your reference dataset.
Use `--no-venus` to disable these overlays. No VENUS-LEVIS dataset is included.

## Interpretation and analysis options

The reference curves include asymptotic slowing-down expressions and
pySTEL's Maxwellian energy collision operator. Both use marker and
background-ion masses read from the current BEAMS3D output. The theory
assumes a uniform, single-ion-species plasma; it does not reproduce the
edge variation of the supplied kinetic profiles.

- Theory defaults are `ne = 1.2e19 m^-3`, `Te = 1000 eV`, `Ti = 100 eV`,
  `ln(Lambda) = 15.84`, and unit beam/background charge numbers.
  Set the `--theory-*` options to match the comparison being made.
- `--theory-power-scale` defaults to `0.001`. It scales per-particle theory
  powers by an effective population and is not a unit conversion. Use `1`
  for per-particle watts, or a population consistent with the marker weights
  when comparing with integrated BEAMS3D power.
- Theory heating fractions are instantaneous ratios. Theory energy changes
  use the initial-rate approximation `(Pe + Pi) * time`, with
  `--theory-time` defaulting to `0.001` seconds. They do not integrate the
  changing particle energy over the trajectory. Maxwellian species powers
  retain their sign, including energy gain by the test particle.
- Measured energy loss uses stored rows 1 and 1000 by default, without time
  rescaling. `--energy-row N` changes the endpoint using one-based indexing;
  choose a theory interval consistent with the actual stored times.
- `--exclude TOKEN` can be repeated to omit additional filenames.
  Missing or invalid measurements are omitted with warnings. A file with
  fewer than the requested stored rows has no energy-loss point.

The script generates comparison figures; it does not apply automated
pass/fail tolerances. For reproducible comparisons, record the STELLOPT
revision, compiler/MPI configuration, rank count, input changes, and
analysis options alongside the results.

Run `python3 make_plots.py --help` for all available options. If pySTEL
cannot be imported or its VMEC reader cannot load LIBSTELL, check
`PYTHONPATH`, `STELLOPT_PATH`, and the shared-library installation. If the
script finds no eligible HDF5 files, check `--data-dir` and confirm that
the BEAMS3D runs completed.

## License

[MIT](LICENSE), copyright (c) 2026 Samuel Lazerson.
