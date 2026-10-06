"""Estimate the France ball's backspin from its measured bat tracking.

Statcast does not publish batted-ball spin, but it does measure the swing (bat
speed, attack angle) and the pitch. Backspin is set by the ball-bat collision,
so with the swing, the pitch, and the measured exit velocity and launch angle we
can reconstruct the collision geometry and bound the spin.

Two readings:

- A physics reading. Decompose the bat-ball relative velocity into the parts
  along and across the collision normal (approximately the launch direction).
  The across-normal (tangential) part is what spins the ball. The full-grip
  (rolling) limit is a hard upper bound; real contacts reach only a fraction of
  it because the bat slides across the ball, which puts a ball like this in the
  ~1500-2500 rpm range reported by Statcast-era collision studies (Nathan). We
  report the geometry and a bounded range, not a false-precision number, because
  we cannot measure the true value.
- An empirical reading. Among 2024+ balls at the same exit velocity and launch
  angle, how does measured carry change with the swing's attack angle? A steeper
  swing (attack angle closer to the launch angle) means less undercut and less
  backspin. This shows the spin effect directly in carry, with no spin number
  required.

Source for the collision framework: Alan M. Nathan, "The Physics of Baseball"
(baseball.physics.illinois.edu), the ball-bat collision model and his batted-ball
spin analyses.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from . import physics as P

BALL_R_M = C.BALL_RADIUS_M
# A uniform sphere reaches 5/7 of the no-slip tangential speed as surface spin.
SPHERE_GRIP = 5.0 / 7.0
# Fraction of the full-grip (rolling) spin a real wood-bat contact achieves.
# Nathan's measurements put batted-ball backspin well below the rolling limit
# because the contact slides; ~0.2-0.3 reproduces the observed rpm range. Used
# only to translate the geometric ceiling into a plausible range, with the range
# carried explicitly rather than a single point.
GRIP_FRACTION_LO, GRIP_FRACTION_HI = 0.18, 0.30


def collision_geometry(
    ev_mph: float = C.FRANCE_EV,
    la_deg: float = C.FRANCE_LA,
    bat_mph: float = C.FRANCE_BAT_SPEED,
    attack_deg: float = C.FRANCE_ATTACK_ANGLE,
    pitch_mph: float = C.FRANCE_PITCH_MPH,
    pitch_descent_deg: float = C.FRANCE_PITCH_DESCENT_DEG,
) -> dict:
    """Reconstruct the ball-bat collision geometry and bound the backspin.

    Works in the vertical plane of the ball's flight: +x toward the outfield
    (the hit direction), +z up. The pitch travels in -x and slightly down; the
    bat moves in +x and up along the attack angle. The outgoing ball direction
    (the launch angle) is taken as the collision normal, and the component of the
    bat-ball relative velocity across that normal drives the backspin.

    Arguments:
        ev_mph: Measured exit velocity.
        la_deg: Measured launch angle.
        bat_mph: Measured bat (barrel) speed at contact.
        attack_deg: Measured attack angle (bat path above horizontal).
        pitch_mph: Pitch speed.
        pitch_descent_deg: Pitch's downward angle at the plate.

    Returns:
        Geometry terms, the full-grip spin ceiling (rpm), and a bounded backspin
        range (rpm) from the grip-fraction band.
    """
    aa = np.radians(attack_deg)
    la = np.radians(la_deg)
    pd_ = np.radians(pitch_descent_deg)

    v_ball_in = np.array([-pitch_mph * np.cos(pd_), -pitch_mph * np.sin(pd_)])
    v_bat = np.array([bat_mph * np.cos(aa), bat_mph * np.sin(aa)])
    v_rel = v_ball_in - v_bat  # ball relative to bat

    n_hat = np.array([np.cos(la), np.sin(la)])  # outgoing ~ along collision normal
    t_hat = np.array([-np.sin(la), np.cos(la)])  # tangential (backspin) direction
    v_rel_n = float(v_rel @ n_hat)
    v_rel_t = float(v_rel @ t_hat)
    obliquity = float(np.degrees(np.arccos(abs(v_rel_n) / np.linalg.norm(v_rel))))

    vt_ms = abs(v_rel_t) * C.MPH_TO_MS
    w_ceiling = SPHERE_GRIP * vt_ms / BALL_R_M  # rad/s, full grip
    rpm_ceiling = w_ceiling * 60 / (2 * np.pi)
    rpm_lo = rpm_ceiling * GRIP_FRACTION_LO
    rpm_hi = rpm_ceiling * GRIP_FRACTION_HI

    return {
        "undercut_deg": round(la_deg - attack_deg, 1),
        "obliquity_deg": round(obliquity, 1),
        "v_rel_mph": round(float(np.linalg.norm(v_rel)), 1),
        "v_rel_tangential_mph": round(abs(v_rel_t), 1),
        "spin_ceiling_rpm": round(float(rpm_ceiling)),
        "spin_estimate_rpm_lo": round(float(rpm_lo), -1),
        "spin_estimate_rpm_hi": round(float(rpm_hi), -1),
    }


def empirical_attack_slope(df, ev_band=(102.0, 108.0), la_band=(46.0, 52.0)) -> dict | None:
    """How measured carry depends on the swing's attack angle, at fixed EV/LA.

    Fits carry ~ attack_angle (ordinary least squares) among 2024+ balls in a
    narrow exit-velocity and launch-angle band around France, so exit velocity
    and launch angle are held roughly fixed and the attack angle (a backspin
    proxy) is what varies.

    Arguments:
        df: Enriched batted-ball table with bat tracking.
        ev_band: Exit-velocity window (mph).
        la_band: Launch-angle window (deg).

    Returns:
        Slope (ft of carry per deg of attack angle), sample size, and the fitted
        carry at France's attack angle versus the band's median attack angle, or
        None if too few balls.
    """
    sub = df[
        (df.get("game_year", 0) >= 2024)
        & df["launch_speed"].between(*ev_band)
        & df["launch_angle"].between(*la_band)
        & df["attack_angle"].notna()
        & df["hit_distance_sc"].notna()
    ]
    sub = sub[sub["hit_distance_sc"].between(0, 520)]
    if len(sub) < 60:
        return None
    x = sub["attack_angle"].to_numpy(float)
    y = sub["hit_distance_sc"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    med_attack = float(np.median(x))
    fit_france = slope * C.FRANCE_ATTACK_ANGLE + intercept
    fit_median = slope * med_attack + intercept
    return {
        "n": int(len(sub)),
        "slope_ft_per_deg": round(float(slope), 2),
        "median_attack_deg": round(med_attack, 1),
        "carry_at_france_attack": round(float(fit_france), 1),
        "carry_at_median_attack": round(float(fit_median), 1),
        "ev_band": ev_band,
        "la_band": la_band,
    }


def _vrel_t_vec(bat_mph, attack_deg, la_deg, pitch_mph=88.0, pitch_desc_deg=6.0):
    """Tangential (backspin-driving) component of the bat-ball relative velocity.

    Uses a fixed league-average pitch so the comparison across balls isolates the
    swing, since public Statcast has no per-ball pitch on the batted-ball row.
    """
    aa = np.radians(np.asarray(attack_deg, float))
    la = np.radians(np.asarray(la_deg, float))
    pd_ = np.radians(pitch_desc_deg)
    rx = -pitch_mph * np.cos(pd_) - np.asarray(bat_mph, float) * np.cos(aa)
    rz = -pitch_mph * np.sin(pd_) - np.asarray(bat_mph, float) * np.sin(aa)
    return np.abs(rx * (-np.sin(la)) + rz * np.cos(la))


def collision_carry_model(df, fit, ev_band=(100.0, 110.0), la_band=(44.0, 54.0)) -> dict | None:
    """Can a ball-bat collision model tighten the France carry band?

    For 2024+ fly balls around France, take the residual between measured carry
    and the physics free-flight carry (at each ball's exit velocity, launch angle,
    and air density), and regress it on the collision spin-driver (the tangential
    relative velocity from the measured swing). If the swing pins the spin, the
    residual scatter should collapse. It barely does, which is the finding: the
    carry spread at fixed launch conditions is dominated by the unmeasured contact
    offset, not by anything bat tracking sees.

    Arguments:
        df: Enriched batted-ball table with bat tracking.
        fit: The fitted physics model.
        ev_band: Exit-velocity window (mph).
        la_band: Launch-angle window (deg).

    Returns:
        Before/after residual std, the variance explained, and France's carry and
        band under the conditioned model, or None if too few balls.
    """
    sub = df[
        (df.get("game_year", 0) >= 2024)
        & df["launch_speed"].between(*ev_band)
        & df["launch_angle"].between(*la_band)
        & df["bat_speed"].notna()
        & df["attack_angle"].notna()
        & df["hit_distance_sc"].between(150, 520)
        & df["air_density"].notna()
    ]
    if len(sub) < 200:
        return None
    ev = sub["launch_speed"].to_numpy(float)
    la = sub["launch_angle"].to_numpy(float)
    phys = P.carry_vec(ev, la, fit.cl(la), sub["air_density"].to_numpy(float), fit.cd(la))
    resid = sub["hit_distance_sc"].to_numpy(float) - phys
    vt = _vrel_t_vec(sub["bat_speed"].to_numpy(float), sub["attack_angle"].to_numpy(float), la)
    b, a = np.polyfit(vt, resid, 1)
    pred = a + b * vt
    std0, std1 = float(resid.std()), float((resid - pred).std())
    r2 = float(1 - (resid - pred).var() / resid.var())

    vt_f = float(_vrel_t_vec(np.array([C.FRANCE_BAT_SPEED]), np.array([C.FRANCE_ATTACK_ANGLE]), np.array([C.FRANCE_LA]))[0])
    rho_dome = P.rho(C.DOME_TEMP_F, C.DOME_ELEV_M, C.DOME_RH)
    phys_f = float(P.carry(C.FRANCE_EV, C.FRANCE_LA, float(fit.cl(C.FRANCE_LA)), rho_dome, cd=float(fit.cd(C.FRANCE_LA))))
    adj_f = float(a + b * vt_f)
    carry_f = phys_f + adj_f
    return {
        "n": int(len(sub)),
        "resid_std_before_ft": round(std0, 1),
        "resid_std_after_ft": round(std1, 1),
        "variance_explained": round(r2, 3),
        "slope_ft_per_mph": round(float(b), 2),
        "france_vrel_t_mph": round(vt_f, 1),
        "pool_mean_vrel_t_mph": round(float(vt.mean()), 1),
        "france_carry_ft": round(carry_f, 1),
        "france_adjustment_ft": round(adj_f, 1),
        "p05_ft": round(carry_f - 1.645 * std1, 1),
        "p95_ft": round(carry_f + 1.645 * std1, 1),
    }


if __name__ == "__main__":  # quick manual check
    print(collision_geometry())
