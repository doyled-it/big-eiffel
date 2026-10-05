# france-roof-ball

**Would Ty France's ball have cleared the fence if it hadn't hit the roof cable?**

In Game 1 of the 2025 NLDS in Milwaukee, with the roof closed, Ty France hit a
ball 105.3 mph at a 49 degree launch angle down the left-field line. It struck a
roof support cable near its apex and was caught on the warning track. This repo
estimates, three independent ways, the probability that the same ball would have
carried over the 344 ft left-field fence had the cable not been there.

## The headline

The ball most likely would not have gone out, but it was far from hopeless.

- **Expected carry was about 320 ft**, roughly 24 ft short of the 344 ft fence.
  The empirical mean is 320 ft, the physics model gives 316 ft, and the ML
  models give 314 to 321 ft.
- There was about a **1-in-8 chance (12.4%, plausibly 9 to 17%)** of carrying
  the raw 344 ft on the ground, measured directly from 250 real balls hit at the
  same speed and angle over 2015 to 2025.
- Across all methods the raw number spans about 7 to 15%, and the two most
  directly grounded estimators, the empirical pool and the physics-anchored ML
  model, both land near 12%.
- That falls to roughly **4 to 7% (very roughly 1 in 15 to 1 in 25)** once you
  also require the ball to clear the 8 ft wall while staying fair, rather than
  just reach the fence distance along the ground. The empirical estimate there
  is 6.8% (Wilson 4 to 11%), and the physics model puts it as low as 3.6%.
- All three methods agree the central carry is about 314 to 321 ft and the clear
  probability is single-digit to low-teens percent.
- The single biggest source of uncertainty is **backspin**, which the public
  Statcast feed does not measure. So the honest output is a distribution over
  carry, and the probability comes from its upper tail.

In one line: most likely a long out, with a real minority chance it was a home
run, and the number is low but not zero.

## The question, precisely

Exit velocity and launch angle are measured at the bat, before the ball travels
anywhere near the roof. Removing the cable therefore changes none of the inputs.
It only lets the ball finish its natural flight. So the question is the standard
projected-distance question: given a ball launched at 105.3 mph and 49 degrees
down the left-field line, how far does it carry, and does that beat the fence
while staying fair?

"Clears" is reported two ways throughout:

- **Raw distance**: the carry reaches the fence distance (344 ft on the line).
- **Full clear**: the ball also stays fair and is still above the 8 ft wall when
  it arrives. A ball that merely lands at the wall base is at zero height there,
  so an 8 ft wall adds a few feet of required carry, set by the descent angle.

The France ball's measured spray angle (about -44 degrees) is fair, one degree
inside the foul line, so fairness does not reduce the probability further for
this specific ball. It matters only through the park geometry.

## Data

Primary source: the HuggingFace dataset
[`Jensen-holm/statcast-era-pitches`](https://huggingface.co/datasets/Jensen-holm/statcast-era-pitches),
a parquet mirror of MLB Statcast pitch-by-pitch data from 2015 to the present,
which in turn comes from Baseball Savant. The single source file is downloaded
once and filtered locally with DuckDB to regular-season batted balls that have a
measured exit velocity, launch angle, and Statcast hit distance. That leaves
**2,363,279 batted balls**. A `pybaseball` season-by-season pull is included as a
fallback.

Spray angle is derived from the Statcast hit coordinates:

```
spray_deg = degrees(atan2(hc_x - 125.42, 198.27 - hc_y))
```

Negative is toward left field, positive toward right field. The sign and scale
check out: right-handed batters pull to negative (a few degrees to the pull side
on hard-hit balls, mean about -7 degrees at 95 mph and up), left-handed batters
to positive, and the foul lines sit near plus or minus 45 degrees.

### Validation gate

These targets were fixed in advance, from an independent pass over the same
Statcast data. The pipeline reproduces them exactly, which confirms the pull,
filter, and carry definition here match the reference (the window can also be
cross-checked with a Baseball Savant search for the same exit velocity and
launch angle):

| metric | target | this repo |
|---|---|---|
| n (104 to 106.5 mph, 48 to 50 deg) | 250 | 250 |
| mean carry | 320.1 ft | 320.1 ft |
| median carry | 320.5 ft | 320.5 ft |
| cleared 344 ft | 31 | 31 |
| fraction | 12.4% | 12.4% |
| Wilson 95% CI | about 9 to 17% | 8.9 to 17.1% |

## The three methods

### Method 1: empirical pooling (the model-free baseline)

Pool every comparable batted ball and read the carry distribution straight off
the data. Reported at the exact gate window and at widening windows around
105.3 / 49, plus a Gaussian-kernel-weighted estimate centered exactly on the
France profile so the estimate is not forced into a bin. This is the most direct
possible evidence: balls like this one, how far did they actually go, none of
which hit a cable.

### Method 2: calibrated drag-plus-Magnus physics

A quadratic-drag, Magnus-lift point-mass integrator with a humidity-aware air
density model. The drag coefficient is fixed; an effective lift coefficient is
fit as a smooth function of launch angle so the model reproduces the empirical
mean-carry surface across a grid of exit-velocity and launch-angle bins in
average MLB conditions, not calibrated at a single point. A single smooth
effective-lift curve fits the whole surface to about 10 ft RMSE and puts the
carry peak near 30 degrees (404 ft in the model, 405 ft in the data). Because
the fit targets that surface, this is a consistency check that the calibration
is physically coherent rather than overfit, not an independent prediction.

It is then evaluated at the France profile under closed-roof dome conditions
(about 72 F, elevation 193 m, no wind), with a sensitivity sweep over
temperature (60 to 80 F) and over a plausible backspin range (lift coefficient
varied by about plus or minus 35%, since spin is unmeasured).

A caveat worth stating plainly: a constant-drag lift-only model reproduces carry
well up to about 35 degrees but structurally overshoots the steep 36 to 44
degree descent regime, where carry versus lift is non-monotone and the effective
lift is poorly identified. The France profile at 49 degrees sits in that hard
regime, so the physics point there carries extra model uncertainty and is
treated as a cross-check, not the primary estimate.

### Method 3: physics-informed machine learning

LightGBM quantile regression (not deep learning, since this is low-dimensional
tabular data), built two ways:

- **Direct**: predict the carry distribution from exit velocity, launch angle,
  spray angle, season, and park, fitting several quantiles to get a predictive
  distribution.
- **Physics-informed residual**: predict the residual between measured carry and
  the physics-model carry, then add it back. This anchors the model in physics
  and lets the trees learn the park, spray, and era corrections.

A 2024+ variant adds the bat-tracking fields (attack angle, swing path tilt, bat
speed, swing length) as a partial spin proxy. All models are checked for
interval calibration on a time-based holdout: the predicted 90, 80, and 50
percent intervals cover almost exactly 90, 80, and 50 percent of held-out
carries, so the predicted distributions are honest.

## Results

All values are P(carry clears the fence). "raw" uses the ground distance;
"+ 8ft wall" additionally requires the ball to be above the 8 ft wall when it
arrives. The France ball's measured spray (about -44 degrees) is fair, so
fairness does not reduce these further. Intervals shown are Wilson 95% for the
empirical counts.

| method | central carry (ft) | generic 344 raw | generic 344 + 8ft wall | Milwaukee park raw | Milwaukee park + 8ft wall |
|---|---|---|---|---|---|
| Method 1 empirical (EV/LA pool, n=250) | 320 | 12.4% [8.9, 17.1] | 9.6% [6.5, 13.9] | 9.6% [6.5, 13.9] | 6.8% [4.3, 10.6] |
| Method 1 kernel (sigma 1.5) | 320 | 14.6% | | | |
| Method 2 physics (sensitivity MC) | 316 | 9.3% | 5.8% | 5.8% | 3.6% |
| Method 3 ML direct | 321 | 7.3% | 5.8% | 5.8% | 5.0% |
| Method 3 ML physics-residual | 321 | 11.5% | 5.0% | 5.0% | 5.0% |
| Method 3 ML + bat-tracking (2024+) | 314 | 6.8% | 5.7% | 5.7% | 5.0% |

The physics central carry (316 ft) is the dome point estimate; its four
probabilities come from the temperature-and-spin Monte Carlo sweep, whose carry
mean is 313 ft.

At the France spray angle the wall is 348 ft, four feet deeper than the 344 ft
foul-line minimum. So reaching the Milwaukee wall on the ground (9.6%) is a
little harder than reaching a flat 344 ft (12.4%), and clearing the 8 ft wall
there brings it to 6.8%. The machine-learning uncertainty is best read as the
spread across the three methods, which lines up with the empirical Wilson
interval. A naive bootstrap would understate it, since resampling thins the
sparse high-angle tail.

Reading across methods, the central carry clusters at 314 to 321 ft and the raw
P(reach 344) spans about 7 to 15%, with the stricter fair-and-over-the-wall
number a few points lower. The empirical pool (12.4%) and the physics-residual
model (11.5%), the two most directly grounded estimators, both land near 12%.

## Figures

All saved to `figures/`:

- `carry_vs_launch_angle.png`: empirical mean carry against the fitted physics
  curve, including the peak near 30 degrees and the harder high-angle regime.
- `france_carry_distribution.png`: the carry distribution of the 250 comparable
  balls, with the 344 ft line and the cleared fraction marked.
- `ml_predictive_distribution.png`: the LightGBM predictive carry CDF at the
  France input, with P(carry >= 344) read off the tail.
- `park_wall_map.png`: a top-down map of American Family Field wall distance by
  spray angle, with the France line marked.

## Limitations

1. **No measured spin (the dominant one).** Public Statcast gives no batted-ball
   backspin or sidespin. Backspin is what turns a given launch into more or less
   carry, and at a 49 degree launch it can push the ball back on the long ascent.
   This is why the output is a distribution, not a point, and why the methods
   disagree by a few points. The bat-tracking fields are only a weak proxy and
   only exist from 2024.
2. **Statcast hit distance is a projection.** `hit_distance_sc` is Statcast's
   modeled flight distance, not a surveyed landing point, and carries its own
   error.
3. **Physics model at high launch angle.** As noted, the lift-only model is
   poorly constrained in the 36 to 44 degree transition, so its France point is a
   cross-check rather than the lead number.
4. **Park geometry is approximate.** The wall distance is interpolated from a
   handful of published anchor dimensions, and the dome's exact temperature and
   air density on the night are assumed, not measured.
5. **Spray is treated as fixed.** The analysis conditions on the measured spray
   of about -44 degrees. It does not model the chance the ball hooks foul, since
   the measured ball was fair.

## Reproduce it

Requires [`uv`](https://docs.astral.sh/uv/). On macOS, LightGBM needs OpenMP:

```bash
brew install libomp
```

Then:

```bash
uv sync                 # install dependencies
make all                # pull data, run all three methods, write tables + figures
```

Or step by step:

```bash
uv run python -m frb.data      # download and cache the batted-ball data
uv run python run.py           # run everything (seeded, deterministic)
uv run pytest                  # unit tests, including the validation gate
```

The first run downloads an 826 MB parquet once (cached under the HuggingFace
cache) and filters it to a 34 MB local parquet (gitignored). A small sample and
all aggregated outputs are committed.

## Layout

```
frb/
  config.py      constants: the event, park geometry, physics, validation gate
  data.py        pull, cache, filter, and the spray-angle formula
  geometry.py    park wall distance and the clear definitions
  empirical.py   Method 1: pooling and the Gaussian-kernel estimate
  physics.py     Method 2: the calibrated drag-plus-Magnus model
  ml.py          Method 3: physics-informed quantile regression
  plots.py       the four figures
  report.py      the cross-method results table
  stats_utils.py Wilson interval, weighted quantile
run.py           single entry point that reproduces everything
tests/           unit tests plus the data-gated validation gate
```

## Sources

- Statcast data: Baseball Savant, via the HuggingFace mirror
  `Jensen-holm/statcast-era-pitches`.
- Baseball flight aerodynamics (drag and Magnus lift) follow the standard
  point-mass treatment used in the baseball physics literature.
- American Family Field dimensions from published park references.
