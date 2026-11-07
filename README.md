# 2026 F1 Energy Deployment Analysis

## Question

Under the 2026 Formula 1 regulations, roughly half of the car's power comes from the electric motor. **Where on a lap do cars run short of electrical energy, and how much speed does that cost?**

This project estimates that from public telemetry, using a simple physics-based model with explicit validation and uncertainty.

## Approach

### 1. Validate the data first
Public telemetry (via [FastF1](https://github.com/theOehrly/Fast-F1)) is low-frequency and contains errors. Before any modelling, automated checks flag:
- timing gaps (missing samples)
- physically impossible speed jumps
- throttle and brake applied at the same time
- distance going backwards, or GPS positions jumping more than the speed allows

Each check is covered by `pytest` tests. The project reports how many samples each check flags, and how much those samples would distort the results if left in.

### 2. Estimate power with physics
On a straight at full throttle, the power used is approximately:

```
P ≈ v · ( m·a  +  ½·ρ·CdA·v²  +  Crr·m·g )
```

where `v` is speed, `a` is acceleration (from the speed trace smoothed with a Savitzky–Golay filter), `m` is mass, `ρ` is air density, `CdA` is drag area and `Crr` is rolling resistance. A drop in estimated power while the driver is still at full throttle indicates the car is energy-limited.

### 3. Control for what we don't know
A car's exact mass and drag are not public, and an error in drag distorts the estimate most at high speed, which is exactly where energy limits appear. To avoid mistaking a bad assumption for a real effect, the analysis **compares each car against itself** (qualifying laps against race laps, and lap against lap), so that drag and mass largely cancel out.

### 4. Quantify uncertainty
A Monte Carlo simulation repeats the calculation across realistic ranges of mass and drag, and across several laps, and results are reported as ranges rather than single numbers.

## Planned structure

```
f1-energy-2026/
├── README.md
├── requirements.txt
├── analysis.ipynb      # the full story, top to bottom
├── src/
│   ├── load.py         # FastF1 data loading
│   ├── validate.py     # data quality checks
│   └── physics.py      # power estimation and uncertainty
├── tests/
│   └── test_validate.py
└── figures/
```

## Known limitations

- No battery or energy-store data is publicly available, so energy use is inferred, not measured.
- Public telemetry is sampled at only a few samples per second, so acceleration must be smoothed.
- Mass, drag and air density are assumed within ranges, not known.

## Author

Suhani Singh · [github.com/Sush893](https://github.com/Sush893)
