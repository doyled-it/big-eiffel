# Big Eiffel

**Would Ty France's ball have cleared the fence if it hadn't hit the roof cable?**

In Game 1 of the 2026 NLDS in Milwaukee, with the roof closed, Ty France hit a
ball 105.3 mph at a 49 degree launch angle down the left-field line. It struck a
roof support cable near its apex and was caught in left field. This repo
estimates, three independent ways, the probability that the same ball would have
carried over the 344 ft left-field fence had the cable not been there.

## The headline

The ball most likely would not have gone out, but the chance it would was real.

- **Expected carry was about 318 to 321 ft**, roughly 25 ft short of the 344 ft
  fence. The empirical mean is 319 ft, the physics model gives 318 ft, and the
  ML models give 320 to 321 ft.
- There was about a **1-in-8 chance (11.8%, plausibly 8 to 16%)** of carrying
  the raw 344 ft on the ground, measured directly from 280 regular-season and postseason balls hit at the
  same speed and angle over 2015 to 2026. The physics model, now fit to each
  ball's actual air density, independently agrees at 10.9%.
- Across methods the raw number spans about 5 to 14%. The two most directly
  grounded estimators, the empirical pool (11.8%) and the physics model (10.9%),
  land at 11 and 12 percent; the machine-learning models, which condition on the exact dome
  air, sit lower, 5 to 7%, with the bat-tracking model that also sees France's
  real swing at about 8.5%.
- That falls to roughly **5 to 7%** once you also require the ball to clear the
  8 ft wall while staying fair, rather than just reach the fence distance along
  the ground. The empirical estimate there is 6.4% (Wilson 4 to 10%).
- Central carry clusters at 318 to 321 ft and the clear probability is
  single-digit to low-teens percent across the board.
- **We found the actual swing.** Statcast bat tracking has France's contact: a
  75.8 mph barrel on a 21.9 degree uppercut, which undercut the ball by about 27
  degrees and put an estimated 1,600 to 2,600 rpm of backspin on it. Fed that
  real swing rather than a league-typical one, the bat-tracking model lands 320
  ft and reaches 344 about 8.5% of the time, a touch more than a typical swing,
  because a steeper swing to the same 49 degrees means slightly less backspin.
- Backspin is still not measured directly (the public Statcast feed does not
  publish batted-ball spin), but for this ball the measured swing bounds it and
  the bat-tracking model conditions on the real contact. The honest output
  remains a distribution over carry, with the probability from its upper tail.

Most likely a long out, with a real minority chance it was a home run.

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

Primary source for 2015 to 2025: the HuggingFace dataset
[`Jensen-holm/statcast-era-pitches`](https://huggingface.co/datasets/Jensen-holm/statcast-era-pitches),
a parquet mirror of MLB Statcast pitch-by-pitch data that comes from Baseball
Savant. The single source file is downloaded once and filtered locally with
DuckDB to regular-season and postseason batted balls that have a measured exit velocity, launch
angle, and Statcast hit distance. The mirror stops at 2025-09-22, so **2026 is
pulled directly from Baseball Savant** through a fast batted-ball-filtered CSV
endpoint and merged. The full pool is **2,522,038 batted balls, 2015 through
2026**.

Every ball is then joined to the conditions it was hit in (roof state, air
density, and wind), described in [The air it was hit in](#the-air-it-was-hit-in).

Spray angle is derived from the Statcast hit coordinates:

```
spray_deg = degrees(atan2(hc_x - 125.42, 198.27 - hc_y))
```

Negative is toward left field, positive toward right field. The sign and scale
check out: right-handed batters pull to negative (a few degrees to the pull side
on hard-hit balls, mean about -7 degrees at 95 mph and up), left-handed batters
to positive, and the foul lines sit near plus or minus 45 degrees.

### Validation gate

These targets were fixed in advance, from an independent pass over the 2015 to
2025 Statcast data. The pipeline reproduces them exactly on the 2015-2025
subset, which confirms the pull, filter, and carry definition match the
reference. The gate stays pinned to 2015-2025 as a checkpoint; the reported
empirical estimate uses the full 2015-2026 pool of regular season plus postseason (n = 280, mean 319.2 ft, 33
cleared, 11.8%, Wilson 8.5 to 16.1%).

| metric (2015-2025 subset) | target | this repo |
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
France profile so the estimate is not forced into a bin. No model stands between
the question and the answer: balls like this one, how far they actually went,
none of which hit a cable.

### Method 2: calibrated drag-plus-Magnus physics

A quadratic-drag, Magnus-lift point-mass integrator with a humidity-aware air
density model. Two effective coefficients are fit as smooth functions of launch
angle: a low, physical lift coefficient, and a drag multiplier that rises at
steep angles where a ball loses more to the air than a fixed drag coefficient
predicts. They are fit across exit velocity, launch angle, and air density (see
below), so the coefficients are free of the park and weather mix. A single pair
of smooth curves fits the whole mean-carry surface to **2.65 ft RMSE** and
tracks the carry-vs-launch-angle curve across its entire range, with no
transition artifact, reproducing the ~402 ft peak near 30 degrees.

The fitted model is then evaluated at the France profile under closed-roof dome
conditions (about 72 F, elevation 193 m, no wind), giving a central carry of
**317.8 ft** (about 26 ft short of 344). The uncertainty comes from a Monte
Carlo over temperature (60 to 80 F) and the unmeasured backspin, whose per-ball
carry scatter (about 23 ft) is measured from comparable balls at near-identical
conditions. That yields a physics P(clear 344) of 10.9%, now in line with the
empirical estimate rather than a weaker cross-check.

An earlier lift-only version of this model overshot the steep 36 to 44 degree
band badly; adding the launch-angle-dependent drag term fixed it.

### Method 3: physics-informed machine learning

LightGBM quantile regression (not deep learning, since this is low-dimensional
tabular data), built two ways:

- **Direct**: predict the carry distribution from exit velocity, launch angle,
  spray angle, air density, along-flight wind, season, and park, fitting several
  quantiles to get a predictive distribution. The weather features mean this is
  the one method that sees the exact conditions: France is predicted at the
  dome's real air density with no wind, which pulls its estimate below the
  all-conditions rate (the comparable balls that reached 344 were often helped
  by warm or thin air the closed roof did not have).
- **Physics-informed residual**: predict the residual between measured carry and
  the physics-model carry, then add it back. This anchors the model in physics
  and lets the trees learn the park, spray, era, and conditions corrections.

A 2024+ variant adds the bat-tracking fields (attack angle, swing path tilt, bat
speed, swing length) and is evaluated at France's **actual measured swing** (see
below), not a league-typical one. All models are checked for interval calibration
on a time-based holdout: the nominal 89, 80, and 50 percent intervals cover about
89, 79, and 49 percent of held-out carries, so the predicted distributions are
honest.

### The swing and the spin

Statcast never publishes a batted ball's spin, but since 2024 it has tracked the
bat, and France's swing is on record (`game_pk` 849830): a **75.8 mph** barrel on
a **21.9 degree** uppercut, against a 98.9 mph fastball. The ball left at 49
degrees, so the bat undercut it by about **27 degrees**, a backspin-heavy contact.
`frb/spin.py` turns that into two readings:

- **A collision estimate.** Decomposing the bat-ball relative velocity along and
  across the launch direction (Nathan's ball-bat collision framework), the
  across-direction component drives the spin. The full-grip rolling limit is a
  hard ceiling (~8,700 rpm); real contacts slide and reach a fraction of it, which
  puts this ball at roughly **1,600 to 2,600 rpm** of backspin. It is a bounded
  estimate, not a measurement.
- **An empirical reading.** Among 2024+ balls at the same exit velocity and launch
  angle, carry rises about **1.5 ft per degree** of attack angle. A steeper swing
  to the same launch means less undercut, so less of the backspin that fights a
  steeply climbing ball. France's 21.9 degree attack is above the comparable
  median (18.7 degrees), so his real swing raises the estimate a little: the
  bat-tracking model lands 320 ft (vs 317 ft for a typical swing) and reaches 344
  about 8.5% of the time. Reaching 49 degrees with a steeper swing takes less
  undercut, so the uppercut did not cost him distance. The 49 degree launch
  itself is why it lands short.
- **Can the swing tighten the band?** Barely. `collision_carry_model` regresses
  the carry residual (measured minus physics) on the collision spin-driver for
  2024+ comparable balls: the swing explains only about **7%** of the carry
  scatter (residual std 20.0 to 19.3 ft), because the spread at a fixed launch is
  set by the exact contact point, which bat tracking does not see. France's
  spin-driver (94.5) is essentially the pool average (96.4), so his backspin was
  about typical for a 49 degree ball despite the dramatic uppercut. The one lever
  that would tighten this is a measured spin, which Hawk-Eye (and TrackMan)
  capture but MLB does not publish.

## The air it was hit in

Every ball is joined to the conditions it was hit in. Per-game roof state and
field-relative wind ("8 mph, Out To CF") come from the MLB StatsAPI across
28,686 games, along with each venue's elevation and orientation. Hourly humidity
and pressure come from the Open-Meteo reanalysis archive at each park, matched to
the game hour. From these, every ball gets a true **air density** (mean about
1.17 kg/m^3, from 0.99 at Coors Field down to about 1.26 in cold sea-level air)
and an **along-flight wind** component. Roof-closed games (16% of balls) are
treated as still, controlled air.

This buys two things.

**Air density, done right.** The physics model is fit across air density as well
as exit velocity and launch angle, so its coefficients no longer absorb the
altitude mix (Coors and the high parks used to inflate them). The same France
ball carries **318 ft in the Milwaukee dome but 344 ft at Coors Field air
density**: enough to reach the 344 ft foul line, though still short of the 348 ft
wall at the France line, so not quite a home run even then. See
`figures/density_effect.png`.

**Wind, measured instead of guessed.** Regressing the carry residual on the
along-flight wind over 150,000 open-air balls, Statcast's projected carry moves
only **0.19 ft per mph of reported tailwind** (standard error 0.01), far below
the ~3 ft per mph a ball feeling the full wind would show. The projected
distance is largely wind-neutralized, which means wind is not a hidden confound
in this analysis, and the closed-roof ball had no wind anyway.

The full enriched dataset (2.52M balls, 27 columns) is published at
[`doyled-it/statcast-batted-balls-weather`](https://huggingface.co/datasets/doyled-it/statcast-batted-balls-weather).

## Results

All values are P(carry clears the fence). "raw" uses the ground distance;
"+ 8ft wall" additionally requires the ball to be above the 8 ft wall when it
arrives. The France ball's measured spray (about -44 degrees) is fair, so
fairness does not reduce these further. Intervals shown are Wilson 95% for the
empirical counts.

| method | central carry (ft) | generic 344 raw | generic 344 + 8ft wall | Milwaukee park raw | Milwaukee park + 8ft wall |
|---|---|---|---|---|---|
| Method 1 empirical (2015-2026 pool, n=280) | 319 | 11.8% [8.5, 16.1] | 8.6% [5.8, 12.4] | 8.6% [5.8, 12.4] | 6.4% [4.1, 9.9] |
| Method 1 kernel (sigma 1.5) | 320 | 14.1% | | | |
| Method 2 physics (sensitivity MC) | 318 | 10.9% | 8.0% | 7.5% | 5.9% |
| Method 3 ML direct | 321 | 6.6% | 5.7% | 5.5% | 5.0% |
| Method 3 ML physics-residual | 321 | 5.0% | 5.0% | 5.0% | 5.0% |
| Method 3 ML + bat-tracking (2024+, real swing) | 320 | 8.5% | 7.6% | 7.4% | 6.5% |

The physics central carry (318 ft) is the dome point estimate; its four
probabilities come from the temperature-and-spin Monte Carlo sweep. The
bat-tracking row now uses France's measured swing (see below), not a typical one.

At the France spray angle the wall is 348 ft, four feet deeper than the 344 ft
foul-line minimum. So reaching the Milwaukee wall on the ground (8.6%) is a
little harder than reaching a flat 344 ft (11.8%), and clearing the 8 ft wall
there brings it to 6.4%. The machine-learning uncertainty is best read as the
spread across the methods; a naive bootstrap would understate it, since
resampling thins the sparse high-angle tail.

Reading across methods, the central carry clusters at 318 to 321 ft and the raw
P(reach 344) spans about 5 to 14%. The empirical pool (11.8%) and the physics
model (10.9%), the two estimators grounded directly in the ball's real
conditions, land at 11 and 12 percent; the machine-learning models sit a few points lower.

## Figures

All saved to `figures/`:

- `trajectory_profile.png`: the headline figure. A to-scale side view from home
  plate of the model's free-flight arc at the dome's air, the 344 ft / 8 ft wall,
  the roof cable marked near the apex, a plausibility band, and the cut-short path.
- `carry_vs_launch_angle.png`: empirical mean carry against the fitted physics
  curve, which now tracks the data across the whole range, peak near 30 degrees
  through the steep decline past 45.
- `france_carry_distribution.png`: the carry distribution of the 280 comparable
  balls, with the 344 ft line and the cleared fraction marked.
- `ml_predictive_distribution.png`: the LightGBM predictive carry CDF at the
  France input, with P(carry >= 344) read off the tail.
- `park_wall_map.png`: a top-down map of American Family Field wall distance by
  spray angle, with the France line marked.
- `density_effect.png`: the France ball's carry across air densities, marking the
  dome, cold sea level, and Coors Field (where it reaches 344, still short of the 348 ft wall).

## Limitations

1. **Backspin is estimated, not measured (the dominant one).** Public Statcast
   gives no batted-ball backspin or sidespin. Backspin is what turns a given
   launch into more or less carry, and at a 49 degree launch it can push the ball
   back on the long ascent. For this ball the measured swing lets us bound it
   (about 1,600 to 2,600 rpm from the collision geometry) and the bat-tracking
   model conditions on the real contact, but a bound is not a measurement. A
   collision model built from the swing tightens the carry band almost not at all
   (the swing explains ~7% of the scatter), so the output is a distribution, not a
   point. The real flight was tracked by Hawk-Eye up to the cable, but those
   positions and any measured spin stay with MLB rather than in the public feed.
   Bat tracking also only exists from 2024 on.
2. **Statcast hit distance is a projection.** `hit_distance_sc` is Statcast's
   modeled flight distance, not a surveyed landing point, and carries its own
   error. The wind analysis suggests it is also largely wind-neutralized.
3. **Reported wind is not field wind.** The MLB wind reading is a station or
   grid value, not the swirling wind inside the stadium bowl, so the measured
   wind response is a lower bound on the true effect. It does not affect the
   France estimate, which is a dome with no wind.
4. **Roof state is best-effort.** Whether a retractable roof was open or closed
   on a given night comes from the MLB game report where available; some games
   may be misclassified.
5. **The dome's exact air on the night is assumed,** not measured (72 F, sea-level
   humidity and pressure at the park's elevation). Park wall distances are
   interpolated from a handful of published anchor dimensions.
6. **Spray is treated as fixed.** The analysis conditions on the measured spray
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
make all                # pull data + weather, run all methods, write tables + figures
```

Or step by step:

```bash
uv run python -m frb.data          # 2015-2025 Statcast
uv run python -m frb.data --2026   # 2026 from Baseball Savant
uv run python run.py               # weather join + all methods (seeded, deterministic)
uv run pytest                      # unit tests, including the validation gate
uv run python scripts/publish_hf.py --dry-run   # build the enriched dataset
```

`run.py` builds the weather join on first use: it downloads an 826 MB Statcast
parquet once, pulls per-game weather and roof from the MLB StatsAPI, and pulls
hourly humidity and pressure per park from Open-Meteo (all cached under `data/`,
gitignored). Small samples and all aggregated outputs are committed.
`scripts/publish_hf.py` publishes the enriched dataset to HuggingFace (needs a
write token).

## Layout

```
frb/
  config.py      constants: the event, park geometry, physics, validation gate
  data.py        pull/cache/filter Statcast (HF 2015-2025, Savant 2026), spray angle
  weather.py     MLB StatsAPI + Open-Meteo join, air density, field-relative wind
  geometry.py    park wall distance and the clear definitions
  empirical.py   Method 1: pooling and the Gaussian-kernel estimate
  physics.py     Method 2: the density-aware drag-plus-Magnus model, wind learning
  ml.py          Method 3: physics-informed quantile regression
  spin.py        backspin from the measured swing (collision + empirical slope)
  plots.py       the six figures
  report.py      the cross-method results table
  stats_utils.py Wilson interval, weighted quantile
run.py           single entry point that reproduces everything
scripts/         chart-data export, site build, HuggingFace dataset publisher
web/             index.template.html: the page, with a __DATA__ placeholder
site/            built static site (index.html) that Cloudflare Pages serves
tests/           unit tests plus the data-gated validation gate
```

## The web page

The reader-facing page lives in `web/index.template.html` (one HTML file, Chart.js
from a CDN, no framework). `make site` injects `outputs/chartdata.json` into it and
writes `site/index.html`, a self-contained static page.

It deploys on [Cloudflare Pages](https://pages.cloudflare.com/):

1. In the Cloudflare dashboard, create a Pages project and connect this GitHub repo.
2. Framework preset **None**, build command **empty**, build output directory **`site`**.
3. Add the custom domain **bigeiffel.doyled-it.com** under the project's Custom Domains.

There is no build step on Cloudflare: `site/index.html` is committed, so every push to
`main` redeploys it. To refresh the page after changing the analysis or the template,
run `make site` and commit the result.

## Sources

- Statcast data: Baseball Savant, via the HuggingFace mirror
  `Jensen-holm/statcast-era-pitches` for 2015 to 2025, and pulled directly from
  Baseball Savant for 2026.
- Game weather, roof state, field-relative wind, and venue elevation and
  orientation: the MLB StatsAPI schedule.
- Hourly humidity and surface pressure: the Open-Meteo historical reanalysis
  archive.
- Baseball flight aerodynamics (drag and Magnus lift) follow the standard
  point-mass treatment used in the baseball physics literature.
- American Family Field dimensions from published park references.
- Enriched dataset:
  [`doyled-it/statcast-batted-balls-weather`](https://huggingface.co/datasets/doyled-it/statcast-batted-balls-weather).
