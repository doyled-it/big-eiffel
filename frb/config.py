"""Shared constants: the event, the park, the physics, and the file paths.

Everything that is a fixed input to the analysis lives here so the three
methods and the plotting code all agree on the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"

for _d in (DATA, FIGURES, OUTPUTS):
    _d.mkdir(exist_ok=True)

# Cached data products.
RAW_PARQUET = DATA / "batted_balls.parquet"  # 2015-2025 HF pull (gitignored)
BALLS_2026_PARQUET = DATA / "batted_balls_2026.parquet"  # 2026 Savant pull (gitignored)
ENRICHED_PARQUET = DATA / "batted_balls_weather.parquet"  # 2015-2026 + weather (gitignored)
MLB_GAMES_PARQUET = DATA / "mlb_games.parquet"  # per-game weather/roof/venue (gitignored)
SAMPLE_PARQUET = DATA / "batted_balls_sample.parquet"  # small committed sample

# Direct Baseball Savant CSV endpoint (faster than day-by-day pybaseball, and
# filters to batted balls server-side).
SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"

# Last date covered by the published base file. The daily updater only appends
# dates after this, so its partitions never duplicate the base. Bump it whenever
# the base is rebuilt and re-published.
DATASET_BASE_LAST_DATE = "2026-10-04"

# Game types kept in the pool: regular season plus the four postseason rounds
# (wild card, division, league championship, world series). Spring training (S),
# exhibition (E), and all-star (A) are excluded. The France analysis filters
# this down to regular season; the published dataset keeps all of these.
GAME_TYPES = ("R", "F", "D", "L", "W")

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------
# HuggingFace mirror of Statcast pitch-by-pitch data, 2015-present.
# Source: https://huggingface.co/datasets/Jensen-holm/statcast-era-pitches
HF_REPO = "Jensen-holm/statcast-era-pitches"
HF_FILE = "data/statcast_era_pitches.parquet"  # single ~826 MB parquet on main
# Auto-converted parquet glob, used only by the direct-httpfs fallback path.
HF_GLOB = "hf://datasets/Jensen-holm/statcast-era-pitches@~parquet/**/*.parquet"

# Columns pulled from the raw parquet. Bat-tracking fields (attack_angle,
# swing_path_tilt, bat_speed, swing_length) only exist from 2024 on.
PULL_COLUMNS = [
    "game_year",
    "game_date",
    "game_pk",
    "sv_id",
    "game_type",
    "home_team",
    "events",
    "bb_type",
    "stand",  # batter handedness (L/R), used to validate the spray sign
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "launch_speed_angle",  # Statcast barrel classification code
    "bat_speed",
    "swing_length",
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
]

# --------------------------------------------------------------------------
# The event: Ty France, NLDS Game 1, 2026-10-03, American Family Field (MIL),
# Brewers vs Padres, bottom 9th. The ball struck a roof support cable with the
# roof closed; the official play is a fly out to left (Padres challenged the
# fair/foul call, upheld). Found in Statcast at game_pk 849830 (see below): a
# 105.3 mph / 49 deg ball down the left-field line, with its bat tracking.
# --------------------------------------------------------------------------
FRANCE_EV = 105.3  # exit velocity, mph
FRANCE_LA = 49.0  # launch angle, deg
FRANCE_SPRAY = -44.0  # spray angle, deg (down the left-field line; negative = LF)
FRANCE_PARK = "MIL"  # Statcast home_team code for the Brewers
FRANCE_GAME_PK = 849830
FRANCE_GAME_DATE = "2026-10-03"

# Measured bat tracking for this exact swing (Statcast, 2024+). A 21.9 deg
# attack angle is a steep uppercut (league average is ~10-12 deg), and the ball
# still launched at 49 deg, so the bat undercut it by ~27 deg: a backspin-heavy
# contact. These are the real inputs to Method 3's bat-tracking model.
FRANCE_BAT_SPEED = 75.8  # mph (barrel speed at contact)
FRANCE_ATTACK_ANGLE = 21.9  # deg (vertical angle of the bat's path at contact)
FRANCE_SWING_PATH_TILT = 34.4  # deg
FRANCE_SWING_LENGTH = 8.3  # ft
FRANCE_PITCH_MPH = 98.9  # the pitch hit: a 4-seam fastball
FRANCE_PITCH_DESCENT_DEG = 3.6  # pitch's downward angle at the plate (from vy0, vz0)
FRANCE_STATCAST_DIST = 323.0  # Statcast hit_distance_sc for the ball (see caveat)

# --------------------------------------------------------------------------
# Statcast hit-coordinate geometry
# --------------------------------------------------------------------------
# Spray angle from the batted-ball landing coordinate (hc_x, hc_y).
#   spray_deg = degrees(atan2(hc_x - HC_X0, HC_Y0 - hc_y))
# Negative is toward left field, positive toward right field, from the
# catcher's perspective. Home plate sits at roughly (125.42, 198.27) in the
# pixel coordinate system Statcast ships.
HC_X0 = 125.42
HC_Y0 = 198.27

# --------------------------------------------------------------------------
# The fence
# --------------------------------------------------------------------------
FENCE_FT = 344.0  # American Family Field left-field foul-line distance
WALL_HEIGHT_FT = 8.0

# American Family Field wall distance as a function of spray angle.
# Anchors run from the left-field foul line (-45 deg) through dead center
# (0 deg) to the right-field foul line (+45 deg). Published park dimensions:
# LF line 344, LF gap ~371, CF ~400, RF gap ~374, RF line 345. We interpolate
# smoothly between these anchors. Fair territory is spray in [-45, +45].
PARK_WALL_ANCHORS = {
    # spray_deg : wall_distance_ft
    -45.0: 344.0,
    -37.5: 371.0,
    -25.0: 390.0,
    0.0: 400.0,
    25.0: 392.0,
    37.5: 374.0,
    45.0: 345.0,
}
FAIR_MIN_DEG = -45.0
FAIR_MAX_DEG = 45.0

# --------------------------------------------------------------------------
# Physics model constants (SI units unless noted)
# --------------------------------------------------------------------------
BALL_MASS_KG = 0.145  # regulation baseball mass
BALL_RADIUS_M = 0.0366  # regulation baseball radius
BALL_AREA_M2 = np.pi * BALL_RADIUS_M**2
GRAVITY = 9.81
CD = 0.35  # drag coefficient (fixed; lift is what we fit)
LAUNCH_HEIGHT_M = 0.9  # contact height above the ground
MPH_TO_MS = 0.44704
M_TO_FT = 3.28084

# Average-MLB atmosphere used when fitting the lift coefficient to the
# empirical mean-carry surface (a reasonable league-wide baseline).
MLB_TEMP_F = 70.0
MLB_ELEV_M = 150.0
MLB_RH = 0.50

# Dome conditions at American Family Field with the roof closed: climate
# controlled, near sea-level-ish elevation for Milwaukee, and no wind.
DOME_TEMP_F = 72.0
DOME_ELEV_M = 193.0
DOME_RH = 0.50

# --------------------------------------------------------------------------
# Validation gates (established externally; the empirical pass must match
# these or there is a data or filter bug).
# --------------------------------------------------------------------------
GATE_EV_LO, GATE_EV_HI = 104.0, 106.5
GATE_LA_LO, GATE_LA_HI = 48.0, 50.0
GATE_N = 250
GATE_MEAN = 320.1
GATE_MEDIAN = 320.5
GATE_CLEAR_344 = 31
