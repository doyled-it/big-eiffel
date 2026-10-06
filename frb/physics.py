"""Method 2: a calibrated drag-plus-Magnus trajectory model.

A quadratic-drag, Magnus-lift point-mass integrator. Two effective coefficients
are fit as smooth functions of launch angle so the model reproduces the
empirical mean-carry surface in average MLB conditions (not calibrated at a
single point):

- an effective lift coefficient Cl(LA), kept low and physical, and
- an effective drag multiplier kd(LA) on the base drag coefficient.

A lift-only, constant-drag model reproduces carry up to about 35 degrees but
structurally overshoots the steep 36 to 44 degree descent regime, where a ball
loses more to the air than a fixed drag coefficient predicts. The effective
drag multiplier absorbs that deficit smoothly, so the lift coefficient stays in
a realistic backspin range and the carry-vs-launch-angle curve matches the data
across the whole range (RMSE about 3 ft). The fitted model is then evaluated at
the France profile under closed-roof dome conditions, with a sensitivity sweep
over temperature and over a plausible (unmeasured) spin range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config as C


def rho(temp_f: float, elev_m: float = C.DOME_ELEV_M, rh: float = 0.55) -> float:
    """Air density (kg/m^3) for temperature, elevation, and humidity.

    Arguments:
        temp_f: Temperature in degrees Fahrenheit.
        elev_m: Elevation in meters.
        rh: Relative humidity (0-1).

    Returns:
        Air density in kg/m^3.
    """
    tk = (temp_f - 32) * 5 / 9 + 273.15
    tc = tk - 273.15
    p = 101325 * (1 - 2.25577e-5 * elev_m) ** 5.2559
    psat = 610.94 * math.exp(17.625 * tc / (tc + 243.04))
    pv = rh * psat
    pd = p - pv
    return pd / (287.05 * tk) + pv / (461.495 * tk)


@dataclass
class Trajectory:
    """A simulated batted-ball flight."""

    x_ft: np.ndarray
    z_ft: np.ndarray
    range_ft: float
    apex_ft: float
    flight_s: float
    descent_angle_deg: float

    def height_at(self, dist_ft: float) -> float:
        """Trajectory height (ft) at a horizontal distance, 0 beyond landing."""
        if dist_ft >= self.range_ft:
            return 0.0
        return float(np.interp(dist_ft, self.x_ft, self.z_ft))


def carry(
    ev_mph: float,
    la_deg: float,
    cl: float,
    rho_: float,
    cd: float = C.CD,
    wind_mph: float = 0.0,
    z0: float = C.LAUNCH_HEIGHT_M,
    dt: float = 0.001,
) -> float:
    """Carry distance in feet (fast path, no trajectory stored).

    Arguments:
        ev_mph: Exit velocity in mph.
        la_deg: Launch angle in degrees.
        cl: Effective lift coefficient.
        rho_: Air density in kg/m^3.
        cd: Effective drag coefficient.
        wind_mph: Along-flight wind (positive is a tailwind that aids carry).
        z0: Contact height in meters.
        dt: Integration time step in seconds.

    Returns:
        Carry (landing) distance in feet.
    """
    drag = 0.5 * rho_ * cd * C.BALL_AREA_M2 / C.BALL_MASS_KG
    lift = 0.5 * rho_ * cl * C.BALL_AREA_M2 / C.BALL_MASS_KG
    w = wind_mph * C.MPH_TO_MS
    v = ev_mph * C.MPH_TO_MS
    th = math.radians(la_deg)
    vx = v * math.cos(th)
    vz = v * math.sin(th)
    x = 0.0
    z = z0
    xp = x
    zp = z
    g = C.GRAVITY
    for _ in range(200000):
        xp, zp = x, z
        svx = vx - w  # airspeed relative to the wind
        sp = math.hypot(svx, vz)
        ax = -drag * sp * svx - lift * sp * vz
        az = -g - drag * sp * vz + lift * sp * svx
        vx += ax * dt
        vz += az * dt
        x += vx * dt
        z += vz * dt
        if z <= 0:
            break
        if x < -1:
            break
    if zp != z:
        frac = zp / (zp - z)
        x = xp + frac * (x - xp)
    return x * C.M_TO_FT


def carry_vec(
    ev_mph, la_deg, cl, rho_, cd, dt: float = 0.004, z0: float = C.LAUNCH_HEIGHT_M, max_steps: int = 3000
) -> np.ndarray:
    """Vectorized no-wind carry for many points at once (feet).

    Integrates every (EV, LA, Cl, rho, Cd) trajectory together as numpy arrays,
    stopping each at its own ground crossing. Much faster than looping the
    scalar integrator, which makes the grid fit tractable.

    Arguments:
        ev_mph: Exit velocities (array).
        la_deg: Launch angles in degrees (array).
        cl: Effective lift coefficients (array).
        rho_: Air densities (array).
        cd: Effective drag coefficients (array).
        dt: Integration time step.
        z0: Contact height in meters.
        max_steps: Safety cap on steps.

    Returns:
        Carry distances in feet (array).
    """
    ev = np.asarray(ev_mph, float)
    th = np.radians(np.asarray(la_deg, float))
    drag = 0.5 * np.asarray(rho_, float) * np.asarray(cd, float) * C.BALL_AREA_M2 / C.BALL_MASS_KG
    lift = 0.5 * np.asarray(rho_, float) * np.asarray(cl, float) * C.BALL_AREA_M2 / C.BALL_MASS_KG
    v = ev * C.MPH_TO_MS
    vx = v * np.cos(th)
    vz = v * np.sin(th)
    n = ev.shape[0]
    x = np.zeros(n)
    z = np.full(n, z0)
    rng = np.zeros(n)
    landed = np.zeros(n, dtype=bool)
    g = C.GRAVITY
    for _ in range(max_steps):
        active = ~landed
        if not active.any():
            break
        sp = np.hypot(vx, vz)
        ax = -drag * sp * vx - lift * sp * vz
        az = -g - drag * sp * vz + lift * sp * vx
        xp, zp = x.copy(), z.copy()
        vx = np.where(active, vx + ax * dt, vx)
        vz = np.where(active, vz + az * dt, vz)
        x = np.where(active, x + vx * dt, x)
        z = np.where(active, z + vz * dt, z)
        newly = active & (z <= 0)
        if newly.any():
            denom = zp[newly] - z[newly]
            frac = np.where(denom != 0, zp[newly] / denom, 1.0)
            rng[newly] = (xp[newly] + frac * (x[newly] - xp[newly])) * C.M_TO_FT
            landed |= newly
    still = ~landed
    rng[still] = x[still] * C.M_TO_FT
    return rng


def post_cable_trajectory(
    ev_mph: float,
    la_deg: float,
    cl: float,
    rho_: float,
    cd: float = C.CD,
    speed_retained: float = 0.78,
    z0: float = C.LAUNCH_HEIGHT_M,
    dt: float = 0.001,
):
    """The ball's path after it clips the cable near its apex.

    Integrates the free flight to the apex, then bleeds a fraction of the
    forward speed to stand in for the glancing cable contact, and integrates the
    rest of the fall with the same drag and backspin lift. The descent therefore
    comes from the real physics: it steepens under gravity, while the backspin
    Magnus force carries it forward and softens that steepening (it does not
    flatten the ball out at the ground). The speed loss is an estimate, so the
    landing point is approximate.

    Arguments:
        ev_mph: Exit velocity (mph).
        la_deg: Launch angle (deg).
        cl: Effective lift coefficient (the backspin lift).
        rho_: Air density (kg/m^3).
        cd: Effective drag coefficient.
        speed_retained: Fraction of forward speed kept through the cable contact.
        z0: Contact height (m).
        dt: Integration step (s).

    Returns:
        (descent_x_ft, descent_z_ft, apex_x_ft, apex_z_ft, landing_ft): the path
        from the apex to the ground, plus the apex and landing in feet.
    """
    drag = 0.5 * rho_ * cd * C.BALL_AREA_M2 / C.BALL_MASS_KG
    lift = 0.5 * rho_ * cl * C.BALL_AREA_M2 / C.BALL_MASS_KG
    v = ev_mph * C.MPH_TO_MS
    th = np.radians(la_deg)
    vx, vz = v * np.cos(th), v * np.sin(th)
    x, z = 0.0, z0
    xs, zs = [x], [z]
    apex_i = 0
    cut = False
    for i in range(200000):
        sp = np.hypot(vx, vz)
        ax = -drag * sp * vx - lift * sp * vz
        az = -C.GRAVITY - drag * sp * vz + lift * sp * vx
        vx += ax * dt
        vz += az * dt
        if not cut and vz <= 0:  # just reached the apex: the cable bleeds forward speed
            apex_i = i + 1
            vx *= speed_retained
            cut = True
        x += vx * dt
        z += vz * dt
        xs.append(x)
        zs.append(z)
        if z <= 0:
            break
    xs = np.array(xs) * C.M_TO_FT
    zs = np.array(zs) * C.M_TO_FT
    if zs[-1] <= 0 and len(zs) > 1:
        z1, z2, x1, x2 = zs[-2], zs[-1], xs[-2], xs[-1]
        frac = z1 / (z1 - z2) if z1 != z2 else 1.0
        land = float(x1 + frac * (x2 - x1))
        xs[-1], zs[-1] = land, 0.0
    else:
        land = float(xs[-1])
    return xs[apex_i:], zs[apex_i:], float(xs[apex_i]), float(zs[apex_i]), land


def integrate(
    ev_mph: float,
    la_deg: float,
    cl: float,
    rho_: float,
    cd: float = C.CD,
    wind_mph: float = 0.0,
    z0: float = C.LAUNCH_HEIGHT_M,
    dt: float = 0.001,
) -> Trajectory:
    """Integrate a full batted-ball trajectory (stores the path).

    Arguments:
        ev_mph: Exit velocity in mph.
        la_deg: Launch angle in degrees.
        cl: Effective lift coefficient.
        rho_: Air density in kg/m^3.
        cd: Effective drag coefficient.
        wind_mph: Along-flight wind (positive is a tailwind that aids carry).
        z0: Contact height in meters.
        dt: Integration time step in seconds.

    Returns:
        A :class:`Trajectory` (distances and heights in feet).
    """
    drag = 0.5 * rho_ * cd * C.BALL_AREA_M2 / C.BALL_MASS_KG
    lift = 0.5 * rho_ * cl * C.BALL_AREA_M2 / C.BALL_MASS_KG
    w = wind_mph * C.MPH_TO_MS
    v = ev_mph * C.MPH_TO_MS
    th = np.radians(la_deg)
    vx = v * np.cos(th)
    vz = v * np.sin(th)
    x, z, t = 0.0, z0, 0.0
    xs, zs = [x], [z]
    for _ in range(200000):
        svx = vx - w
        sp = np.hypot(svx, vz)
        ax = -drag * sp * svx - lift * sp * vz
        az = -C.GRAVITY - drag * sp * vz + lift * sp * svx
        vx += ax * dt
        vz += az * dt
        x += vx * dt
        z += vz * dt
        t += dt
        xs.append(x)
        zs.append(z)
        if z <= 0 or x < -1:
            break
    xs = np.array(xs) * C.M_TO_FT
    zs = np.array(zs) * C.M_TO_FT
    if zs[-1] <= 0 and len(zs) > 1:
        z1, z2 = zs[-2], zs[-1]
        x1, x2 = xs[-2], xs[-1]
        frac = z1 / (z1 - z2) if z1 != z2 else 1.0
        range_ft = float(x1 + frac * (x2 - x1))
    else:
        range_ft = float(xs[-1])
    descent = float(np.degrees(np.arctan2(-vz, vx)))
    return Trajectory(
        x_ft=xs, z_ft=zs, range_ft=range_ft, apex_ft=float(zs.max()), flight_s=t, descent_angle_deg=descent
    )


# --------------------------------------------------------------------------
# Calibrating the effective coefficients against the mean-carry surface
# --------------------------------------------------------------------------

CL_KNOTS_LA = np.array([10.0, 14.0, 18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0, 46.0, 50.0])


@dataclass
class LiftFit:
    """A fitted effective-lift and effective-drag model (functions of LA)."""

    knot_la: np.ndarray
    knot_cl: np.ndarray
    knot_kd: np.ndarray  # drag multiplier on the base CD
    rho_fit: float  # pool mean air density (density-aware fit) or MLB baseline
    grid_pts: np.ndarray
    grid_obs: np.ndarray
    grid_model: np.ndarray
    grid_n: np.ndarray
    grid_rho: np.ndarray = None  # per-bin air density (density-aware fit)
    density_aware: bool = False

    def _cl_interp(self) -> PchipInterpolator:
        return PchipInterpolator(self.knot_la, self.knot_cl, extrapolate=True)

    def _kd_interp(self) -> PchipInterpolator:
        return PchipInterpolator(self.knot_la, self.knot_kd, extrapolate=True)

    def cl(self, la_deg: float | np.ndarray) -> np.ndarray:
        """Effective lift coefficient at a launch angle (clamped to knot range)."""
        la = np.clip(np.asarray(la_deg, dtype=float), self.knot_la.min(), self.knot_la.max())
        return self._cl_interp()(la)

    def kd(self, la_deg: float | np.ndarray) -> np.ndarray:
        """Effective drag multiplier at a launch angle (clamped to knot range)."""
        la = np.clip(np.asarray(la_deg, dtype=float), self.knot_la.min(), self.knot_la.max())
        return self._kd_interp()(la)

    def cd(self, la_deg: float | np.ndarray) -> np.ndarray:
        """Effective drag coefficient = base CD times the fitted multiplier."""
        return C.CD * self.kd(la_deg)

    def rmse(self) -> float:
        """Count-weighted RMSE of model vs empirical mean carry over the grid."""
        w = self.grid_n / self.grid_n.sum()
        return float(np.sqrt(np.sum(w * (self.grid_model - self.grid_obs) ** 2)))


def mean_carry_grid(
    df,
    ev_centers=None,
    la_centers=None,
    half_ev: float = 1.5,
    half_la: float = 1.5,
    min_n: int = 30,
):
    """Empirical mean carry on an (EV, LA) grid.

    Arguments:
        df: Batted-ball table.
        ev_centers: Exit-velocity bin centers (defaults to GRID_EV_CENTERS).
        la_centers: Launch-angle bin centers (defaults to GRID_LA_CENTERS).
        half_ev: Half-width of each EV bin.
        half_la: Half-width of each LA bin.
        min_n: Minimum count for a bin to be used.

    Returns:
        (points, means, counts) arrays.
    """
    ev_centers = GRID_EV_CENTERS if ev_centers is None else ev_centers
    la_centers = GRID_LA_CENTERS if la_centers is None else la_centers
    ev = df.launch_speed.to_numpy(dtype=float)
    la = df.launch_angle.to_numpy(dtype=float)
    d = df.hit_distance_sc.to_numpy(dtype=float)
    pts, means, counts = [], [], []
    for e in ev_centers:
        for a in la_centers:
            m = (np.abs(ev - e) <= half_ev) & (np.abs(la - a) <= half_la)
            n = int(m.sum())
            if n >= min_n:
                pts.append((e, a))
                means.append(float(d[m].mean()))
                counts.append(n)
    return np.array(pts), np.array(means), np.array(counts)


GRID_EV_CENTERS = np.arange(92, 112, 3.0)
GRID_LA_CENTERS = np.arange(10, 51, 2.0)
# Air-density bin edges for the density-aware fit (thin Coors air through dense cold air).
GRID_RHO_EDGES = np.array([1.00, 1.10, 1.16, 1.20, 1.30])

# Fit bounds and priors.
CL_BOUNDS = (0.05, 0.28)  # realistic backspin-lift range
KD_BOUNDS = (0.9, 2.2)  # drag multiplier on base CD


def mean_carry_grid_rho(df, ev_centers=None, la_centers=None, half_ev=1.5, half_la=1.5, min_n=30):
    """Empirical mean carry on an (EV, LA, air-density) grid.

    Binning by air density as well as launch conditions lets the fit separate
    the aerodynamic coefficients from the park-and-weather mix, since a thin-air
    (Coors) bin and a dense-air bin at the same EV and LA must be matched by the
    same coefficients at their own densities.

    Arguments:
        df: Batted-ball table with an ``air_density`` column.
        ev_centers: Exit-velocity bin centers.
        la_centers: Launch-angle bin centers.
        half_ev: Half-width of each EV bin.
        half_la: Half-width of each LA bin.
        min_n: Minimum count for a bin to be used.

    Returns:
        (ev, la, rho, mean_carry, counts) arrays.
    """
    ev_centers = GRID_EV_CENTERS if ev_centers is None else ev_centers
    la_centers = GRID_LA_CENTERS if la_centers is None else la_centers
    ev = df.launch_speed.to_numpy(float)
    la = df.launch_angle.to_numpy(float)
    d = df.hit_distance_sc.to_numpy(float)
    rho_ = df.air_density.to_numpy(float)
    ok = np.isfinite(rho_)
    rows = []
    for e in ev_centers:
        for a in la_centers:
            base = ok & (np.abs(ev - e) <= half_ev) & (np.abs(la - a) <= half_la)
            for i in range(len(GRID_RHO_EDGES) - 1):
                lo, hi = GRID_RHO_EDGES[i], GRID_RHO_EDGES[i + 1]
                m = base & (rho_ >= lo) & (rho_ < hi)
                n = int(m.sum())
                if n >= min_n:
                    rows.append((e, a, float(rho_[m].mean()), float(d[m].mean()), n))
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def fit_lift_curve(df, smooth_cl: float = 2.0e4, smooth_kd: float = 6.0e6, dt: float = 0.004) -> LiftFit:
    """Fit smooth effective Cl(LA) and drag kd(LA) to the mean-carry surface.

    Both coefficients are parameterized by their values at launch-angle knots
    and optimized jointly to minimize the count-weighted squared error between
    model carry and empirical mean carry, with curvature penalties for
    smoothness and a gentle prior that the drag multiplier stays at least one.
    The lift is bounded to a realistic range so the drag multiplier, not an
    unphysical lift, carries the high-angle correction.

    When the table has per-ball air density (the weather-enriched pool), the fit
    is density aware: it bins by (EV, LA, density) and evaluates the model at
    each bin's own density, so the coefficients are free of the altitude and
    weather mix. Otherwise it fits at an average-MLB atmosphere.

    Arguments:
        df: Batted-ball table.
        smooth_cl: Curvature penalty for the lift curve.
        smooth_kd: Curvature penalty for the drag curve.
        dt: Integration step used during fitting (coarser for speed).

    Returns:
        A :class:`LiftFit`.
    """
    from scipy.optimize import minimize

    density_aware = "air_density" in df.columns and bool(df["air_density"].notna().any())
    if density_aware:
        evs, las, rhos, obs, counts = mean_carry_grid_rho(df)
        rho_ref = float(np.average(rhos, weights=counts))
    else:
        rho_ref = rho(C.MLB_TEMP_F, C.MLB_ELEV_M, C.MLB_RH)
        pts, obs, counts = mean_carry_grid(df)
        evs, las = pts[:, 0], pts[:, 1]
        rhos = np.full_like(evs, rho_ref)
    w = np.sqrt(counts.astype(float))
    nk = CL_KNOTS_LA.size

    def model_carry(params, step):
        cl_k, kd_k = params[:nk], params[nk:]
        cli = PchipInterpolator(CL_KNOTS_LA, cl_k, extrapolate=True)
        kdi = PchipInterpolator(CL_KNOTS_LA, kd_k, extrapolate=True)
        cls = np.clip(cli(las), 0.0, None)
        cds = C.CD * np.clip(kdi(las), 0.05, None)
        return carry_vec(evs, las, cls, rhos, cds, dt=step)

    def objective(params):
        cl_k, kd_k = params[:nk], params[nk:]
        model = model_carry(params, dt)
        data_term = np.sum(w * (model - obs) ** 2)
        pen = smooth_cl * np.sum(np.diff(cl_k, 2) ** 2) + smooth_kd * np.sum(np.diff(kd_k, 2) ** 2)
        pen += 1.0e5 * np.sum(np.clip(1.0 - kd_k, 0, None) ** 2)  # prefer kd >= 1
        return data_term + pen

    x0 = np.concatenate([np.full(nk, 0.12), np.full(nk, 1.05)])
    bounds = [CL_BOUNDS] * nk + [KD_BOUNDS] * nk
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-9})
    cl_k, kd_k = res.x[:nk], res.x[nk:]
    grid_model = model_carry(res.x, 0.002)
    return LiftFit(
        knot_la=CL_KNOTS_LA,
        knot_cl=cl_k,
        knot_kd=kd_k,
        rho_fit=rho_ref,
        grid_pts=np.column_stack([evs, las]),
        grid_obs=obs,
        grid_model=grid_model,
        grid_n=counts,
        grid_rho=rhos,
        density_aware=density_aware,
    )


def carry_vs_la(ev_mph: float, fit: LiftFit, rho_: float, la_values: np.ndarray, dt: float = 0.001) -> np.ndarray:
    """Model carry across launch angles at a fixed exit velocity."""
    return np.array([carry(ev_mph, a, float(fit.cl(a)), rho_, cd=float(fit.cd(a)), dt=dt) for a in la_values])


# --------------------------------------------------------------------------
# Evaluating at the France profile
# --------------------------------------------------------------------------


@dataclass
class PhysicsPoint:
    """Physics-model result at a single (EV, LA) point and atmosphere."""

    ev: float
    la: float
    cl: float
    cd: float
    temp_f: float
    rho: float
    carry_ft: float
    apex_ft: float
    descent_deg: float
    shortfall_ft: float


def evaluate_point(
    fit: LiftFit,
    ev: float,
    la: float,
    temp_f: float,
    elev_m: float,
    rh: float,
    cl_scale: float = 1.0,
    fence_ft: float = C.FENCE_FT,
) -> PhysicsPoint:
    """Evaluate the fitted model at a point under given atmosphere.

    Arguments:
        fit: The fitted model.
        ev: Exit velocity (mph).
        la: Launch angle (deg).
        temp_f: Temperature (F).
        elev_m: Elevation (m).
        rh: Relative humidity (0-1).
        cl_scale: Multiplier on the fitted Cl (spin sensitivity).
        fence_ft: Fence distance for the shortfall.

    Returns:
        A :class:`PhysicsPoint`.
    """
    rho_ = rho(temp_f, elev_m, rh)
    cl = float(fit.cl(la)) * cl_scale
    cd = float(fit.cd(la))
    tr = integrate(ev, la, cl, rho_, cd=cd)
    return PhysicsPoint(
        ev=ev,
        la=la,
        cl=cl,
        cd=cd,
        temp_f=temp_f,
        rho=rho_,
        carry_ft=tr.range_ft,
        apex_ft=tr.apex_ft,
        descent_deg=tr.descent_angle_deg,
        shortfall_ft=fence_ft - tr.range_ft,
    )


def sensitivity_mc(
    fit: LiftFit,
    ev: float = C.FRANCE_EV,
    la: float = C.FRANCE_LA,
    elev_m: float = C.DOME_ELEV_M,
    rh: float = C.DOME_RH,
    temp_lo: float = 60.0,
    temp_hi: float = 80.0,
    cl_rel: float = 0.35,
    spin_sigma_ft: float = 20.0,
    n: int = 4000,
    seed: int = 0,
) -> dict:
    """Crude Monte Carlo carry distribution from temperature and spin scatter.

    Temperature is sampled uniformly over a plausible dome range. The unmeasured
    backspin enters two ways: the effective lift is scaled by a normal factor
    whose 2-sigma spans +/- ``cl_rel``, and because the drag-calibrated model
    makes lift only a weak lever at a steep launch angle, the dominant per-ball
    spin scatter is added as Gaussian noise with standard deviation
    ``spin_sigma_ft``, measured from the carry spread of comparable balls at
    near-identical conditions. The effective drag is held at its fitted value.

    Arguments:
        fit: The fitted model.
        ev: Exit velocity (mph).
        la: Launch angle (deg).
        elev_m: Elevation (m).
        rh: Relative humidity.
        temp_lo: Low temperature (F).
        temp_hi: High temperature (F).
        cl_rel: Relative spin range (2-sigma).
        n: Number of Monte Carlo draws.
        seed: RNG seed.

    Returns:
        Dict with the carry samples and summary statistics.
    """
    rng = np.random.default_rng(seed)
    temps = rng.uniform(temp_lo, temp_hi, n)
    cl_scales = rng.normal(1.0, cl_rel / 2.0, n).clip(0.3, 2.0)
    cl0 = float(fit.cl(la))
    cd0 = float(fit.cd(la))
    rhos = np.array([rho(float(t), elev_m, rh) for t in temps])
    carries = carry_vec(np.full(n, ev), np.full(n, la), cl0 * cl_scales, rhos, np.full(n, cd0))
    carries = carries + rng.normal(0.0, spin_sigma_ft, n)
    base = integrate(ev, la, cl0, rho(float(np.mean([temp_lo, temp_hi])), elev_m, rh), cd=cd0)
    return {
        "carries": carries,
        "cl0": cl0,
        "cd0": cd0,
        "mean": float(carries.mean()),
        "median": float(np.median(carries)),
        "p05": float(np.percentile(carries, 5)),
        "p95": float(np.percentile(carries, 95)),
        "descent_deg": base.descent_angle_deg,
    }


# --------------------------------------------------------------------------
# Learning the wind effect from the data
# --------------------------------------------------------------------------


def _carry_surface(fit: LiftFit, dt: float = 0.003):
    """A fast (EV, LA, rho) -> no-wind carry interpolator for the fitted model."""
    from scipy.interpolate import RegularGridInterpolator

    ev_g = np.arange(70.0, 118.0, 2.0)
    la_g = np.arange(0.0, 60.0, 2.0)
    rho_g = np.arange(1.00, 1.31, 0.03)
    EE, AA, RR = np.meshgrid(ev_g, la_g, rho_g, indexing="ij")
    cl = fit.cl(AA.ravel())
    cd = fit.cd(AA.ravel())
    z = carry_vec(EE.ravel(), AA.ravel(), cl, RR.ravel(), cd, dt=dt).reshape(EE.shape)
    return RegularGridInterpolator((ev_g, la_g, rho_g), z, bounds_error=False, fill_value=None)


def learn_wind_effect(
    df, fit: LiftFit, la_lo: float = 15.0, la_hi: float = 45.0, ev_lo: float = 90.0, max_n: int = 150000, seed: int = 0
) -> dict:
    """Empirically measure how much carry responds to wind.

    For open-air balls with meaningful hang time, take the residual between the
    measured carry and the model's no-wind carry at that ball's exit velocity,
    launch angle, and air density, then regress it on the along-flight wind. The
    slope is feet of carry per mph of MLB-reported tailwind. Comparing it to the
    model's own wind sensitivity (feet per mph of true field wind) recovers the
    fraction of the reported wind that the ball actually feels at field level.

    Arguments:
        df: Weather-enriched batted-ball table.
        fit: The fitted physics model.
        la_lo: Lowest launch angle to include.
        la_hi: Highest launch angle to include.
        ev_lo: Lowest exit velocity to include.
        max_n: Cap on the regression sample size.
        seed: RNG seed for subsampling.

    Returns:
        Dict with the fitted slope, standard error, sample size, the model's
        field-wind sensitivity, and the implied effective-wind fraction.
    """
    need = [
        "air_density",
        "wind_along_mph",
        "roof_closed",
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "spray_deg",
    ]
    if any(c not in df.columns for c in need):
        return {"available": False}
    sub = df[
        (~df["roof_closed"].fillna(False))
        & df["air_density"].notna()
        & df["wind_along_mph"].notna()
        & df["launch_angle"].between(la_lo, la_hi)
        & (df["launch_speed"] >= ev_lo)
        & df["spray_deg"].between(-45, 45)
    ]
    if len(sub) < 5000:
        return {"available": False, "n": int(len(sub))}
    if len(sub) > max_n:
        sub = sub.sample(n=max_n, random_state=seed)

    surf = _carry_surface(fit)
    pts = np.column_stack(
        [sub.launch_speed.to_numpy(float), sub.launch_angle.to_numpy(float), sub.air_density.to_numpy(float)]
    )
    model_nowind = surf(pts)
    resid = sub.hit_distance_sc.to_numpy(float) - model_nowind
    wind = sub.wind_along_mph.to_numpy(float)
    good = np.isfinite(resid) & np.isfinite(wind)
    resid, wind = resid[good], wind[good]

    # Ordinary least squares: resid = a + b * wind.
    b, a = np.polyfit(wind, resid, 1)
    yhat = a + b * wind
    n = wind.size
    se = float(np.sqrt(np.sum((resid - yhat) ** 2) / (n - 2) / np.sum((wind - wind.mean()) ** 2)))

    # Model's own field-wind sensitivity at a representative fly-ball profile.
    rr = float(fit.rho_fit)
    cl28, cd28 = float(fit.cl(28)), float(fit.cd(28))
    phys = (carry(100, 28, cl28, rr, cd=cd28, wind_mph=5) - carry(100, 28, cl28, rr, cd=cd28, wind_mph=-5)) / 10.0

    return {
        "available": True,
        "n": int(n),
        "slope_ft_per_mph": float(b),
        "slope_se": se,
        "intercept_ft": float(a),
        "model_field_sensitivity_ft_per_mph": float(phys),
        "effective_wind_fraction": float(b / phys) if phys else float("nan"),
    }
