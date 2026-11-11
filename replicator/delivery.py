"""Full delivery sequence for one droplet:

    jet -> ballistic rise with a co-moving centring trap (acquire) -> park at apex
        -> curved transit around the part -> short hold -> press onto the site.
"""
from dataclasses import dataclass
import numpy as np

from . import constants as C
from .gorkov import Droplet
from .dynamics import Curve, Keys, Planner, concat, drag_force, min_jerk
from .configs import JET_XY


# Final machine (Mk2): the jet sits in the stand under the bottom port, on the axis.
# The nozzle tip is 82 mm below the build centre, just inside the bottom port of the frame;
# the light gate straddles the axis 4 mm above it.  cad.Machine uses the same numbers.
JET_MK2 = dict(xy=(0.0, 0.0), z_acq=-46e-3, z_park=-20e-3, z_nozzle=-82e-3)


@dataclass
class Site:
    xs: np.ndarray        # point on the surface
    n: np.ndarray         # approach axis (unit, pointing away from the part)
    n_surf: np.ndarray = None   # outward surface normal if the approach is oblique

    @property
    def touchdown(self):
        return self.n if self.n_surf is None else self.n_surf


@dataclass
class Jet:
    """Droplet-on-demand jet firing straight up through the bottom board."""
    xy: tuple = JET_XY
    z_acq: float = -42e-3          # height where the co-moving trap takes over
    z_park: float = -20e-3         # nominal apex / parking height
    speed_sigma: float = 0.01      # 1-sigma relative jitter of exit speed
    angle_sigma: float = np.radians(0.3)
    z_nozzle: float = -84e-3       # nozzle tip sits behind the bottom board (see hardware model)

    def nominal(self, drop: Droplet, dt=2e-4):
        """Nominal ballistic rise from z_acq to the apex at z_park (with drag).
        Returns t, x(t), v(t), a(t)."""
        # find the speed at z_acq that puts the apex at z_park
        def rise(v0):
            z, v, t = self.z_acq, v0, 0.0
            zs, vs, ts, as_ = [z], [v], [t], []
            while v > 0:
                a = -C.G_ACC + drag_force(drop, np.array([0, 0, v]))[2] / drop.mass
                as_.append(a)
                z += v * dt + 0.5 * a * dt * dt; v += a * dt; t += dt
                zs.append(z); vs.append(v); ts.append(t)
            as_.append(-C.G_ACC)
            return np.array(ts), np.array(zs), np.array(vs), np.array(as_)
        lo, hi = 0.1, 3.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if rise(mid)[1][-1] < self.z_park:
                lo = mid
            else:
                hi = mid
        t, z, v, a = rise(0.5 * (lo + hi))
        x = np.stack([np.full_like(z, self.xy[0]), np.full_like(z, self.xy[1]), z], 1)
        vv = np.stack([0 * v, 0 * v, v], 1)
        aa = np.stack([0 * a, 0 * a, a], 1)
        return t, x, vv, aa

    def state(self, drop, speed_err=0.0, angle=(0.0, 0.0), time_err=0.0):
        """Droplet state at controller time 0 for a shot whose exit speed is off by
        the fraction speed_err, tilted by `angle` (rad, x and y), and arriving
        time_err seconds late relative to what the controller believes."""
        _, x, v, _ = self.nominal(drop)
        v_acq = v[0, 2]
        h = self.z_acq - self.z_nozzle
        v_exit = np.sqrt(v_acq ** 2 + 2 * C.G_ACC * h)          # drag neglected for the jitter map
        dv_exit = v_exit * speed_err
        dv_acq = v_exit * dv_exit / v_acq                          # from v_acq^2 = v_exit^2 - 2 g h
        dt_arr = dv_exit * (1.0 - v_exit / v_acq) / C.G_ACC + time_err
        ang = np.asarray(angle, float)
        v0 = np.array([ang[0] * v_acq, ang[1] * v_acq, v_acq + dv_acq])
        x0 = x[0] + np.array([ang[0] * h, ang[1] * h, 0.0]) - v0 * dt_arr
        return x0, v0

    def sample(self, drop, rng, gate=True):
        """One random shot.  With the time-of-flight light gate the controller
        re-times the hologram schedule to the measured droplet, leaving ~10 % of
        the speed jitter and 20 us of timing error; without it the full
        exit-speed jitter (and the arrival-time error it causes) remains."""
        e = self.speed_sigma * rng.normal()
        ang = self.angle_sigma * rng.normal(size=2)
        if gate:
            # speed error mostly measured out; arrival time known to the gate's resolution
            _, x, v, _ = self.nominal(drop)
            v_acq = v[0, 2]
            h = self.z_acq - self.z_nozzle
            v_exit = np.sqrt(v_acq ** 2 + 2 * C.G_ACC * h)
            dv_acq = 0.1 * v_exit * (v_exit * e) / v_acq
            v0 = np.array([ang[0] * v_acq, ang[1] * v_acq, v_acq + dv_acq])
            x0 = x[0] + np.array([ang[0] * h, ang[1] * h, 0.0]) - v0 * (20e-6 * rng.normal())
            return x0, v0
        return self.state(drop, e, ang)


def route(park, site: Site, drop: Droplet, part, center, radius, gap0=1.2e-3,
          clearance=5e-3, straight=1.5e-3, blend=7e-3):
    """Smooth collision-free path from the parking point to the stand-off point.

    1. spiral around a bounding sphere (centre, radius + clearance) until above the site,
    2. leave the sphere on a cubic Bezier that turns onto the site normal,
    3. finish with `straight` metres exactly along -n.
    Tangent-continuous everywhere, so the feed-forward acceleration stays finite.
    """
    park = np.asarray(park, float)
    x_pre = site.xs + site.touchdown * drop.a + site.n * gap0
    p_s = x_pre + site.n * straight
    Rs = radius + clearance
    oc = site.xs - center
    b = oc @ site.n
    t_b = -b + np.sqrt(max(b * b - (oc @ oc - Rs * Rs), 0.0))
    p_b = site.xs + site.n * max(t_b, drop.a + gap0 + straight + 2e-3)
    ra, rb = np.linalg.norm(park - center), np.linalg.norm(p_b - center)
    ua, ub = (park - center) / ra, (p_b - center) / rb
    ang = np.arccos(np.clip(ua @ ub, -1, 1))
    arc_len = ang * 0.5 * (ra + rb)
    if arc_len < 1.5 * blend:
        # short hop: one Bezier straight onto the normal
        main = Curve.bezier(park, p_s, dir_in=-site.n, pull=0.45)
        P = main.p
    else:
        f = np.linspace(0, 1, 600)[:, None]
        u = (np.sin((1 - f) * ang) * ua + np.sin(f * ang) * ub) / np.sin(ang)
        r = np.maximum(ra + (rb - ra) * (3 * f ** 2 - 2 * f ** 3), min(ra, rb))
        arc = Curve(center + r * u)
        s_cut = arc.L - blend
        keep = arc.p[arc.s <= s_cut]
        p_c, t_c = arc.pos(s_cut), arc.tangent(s_cut)
        bz = Curve.bezier(p_c, p_s, dir_out=t_c, dir_in=-site.n, pull=0.45)
        P = np.vstack([keep, bz.p[1:]])
    P = np.vstack([P, np.linspace(p_s, x_pre, 60)[1:]])
    curve = Curve(P)
    n_chk = int(len(curve.p) * (1 - 1.2 * straight / curve.L))
    clear = float(part.dist(curve.p[:n_chk]).min()) if part is not None else np.inf
    return curve, x_pre, clear


def plan_delivery(pl: Planner, jet: Jet, site: Site, part, center, radius, gap0=1.2e-3,
                  push=1.0, hold_s=0.02, verbose=False):
    """Hologram keyframes for acquire -> transit -> hold -> press.  Returns Keys + summary."""
    drop = pl.drop
    sv = pl.solver
    # --- acquire: co-moving trap along the nominal ballistic rise, zero acoustic force
    t, x, v, a = jet.nominal(drop)
    nk = max(4, int(np.ceil((x[-1, 2] - x[0, 2]) / (2 * pl.ds))) + 1)
    idx = np.unique(np.round(np.interp(np.linspace(x[0, 2], x[-1, 2], nk), x[:, 2], np.arange(len(t)))).astype(int))
    sv.set_mode()
    phis, amps, infos = [], [], []
    phi, amp = None, 0.5
    for i, j in enumerate(idx):
        ff = (a[j] - drag_force(drop, v[j]) / drop.mass) / C.G_ACC      # = -z_hat: free flight
        phi, amp, info = pl._solve(x[j], ff, phi, amp, fresh=(i == 0))
        phis.append(phi); amps.append(amp); infos.append(info)
    k_acq = Keys(t[idx], np.array(phis), np.array(amps), x[idx], infos)
    park = x[-1]
    # --- transit
    curve, x_pre, clear = route(park, site, drop, part, center, radius, gap0)
    k_tr = pl.transit(curve, phi0=phi, s0=amp)
    k_hold = pl.hold(x_pre, hold_s, k_tr.phi[-1], k_tr.amp[-1])
    k_pr = pl.press(site.xs, site.n, gap0, push=push, phi0=k_hold.phi[-1], n_surf=site.n_surf,
                    gaps=tuple(np.linspace(0.03e-3, gap0, max(5, int(round(gap0 / 0.3e-3)) + 1))))
    keys = concat(concat(concat(k_acq, k_tr), k_hold), k_pr)
    fun = k_pr.info[0]
    kap = np.array([i["kappa"] for i in k_acq.info + k_tr.info + k_hold.info])
    summary = dict(
        t_acq=k_acq.t[-1] - k_acq.t[0], t_transit=k_tr.t[-1] - k_tr.t[0],
        t_press=k_pr.t[-1] - k_pr.t[0], t_total=keys.t[-1] - keys.t[0],
        path_mm=curve.L * 1e3, clearance_mm=clear * 1e3,
        kappa_min=float(kap.min()), kappa_med=float(np.median(kap[:, 0])),
        kappa_press_min=float(fun["kappa_t"].min()), funnel_fn_max=float(fun["f_n"].max()),
        ba_max=float(max(i["Ba"] for i in keys.info)),
        fres_max=float(max(np.linalg.norm(i["f_res"]) for i in k_acq.info + k_tr.info + k_hold.info)),
        amp_mean=float(keys.amp.mean()), n_keys=len(keys.t))
    return keys, summary


def plan_direct(pl: Planner, jet: Jet, site: Site, part, gap0=1.2e-3, push=1.0, hold_s=0.004):
    """Fast delivery for sites the jet can see from below ("direct ascent").

    The droplet is never parked.  From the acquisition height it rises under a
    uniform deceleration chosen so that its velocity reaches zero exactly at the
    stand-off point under the site, while a minimum-jerk lateral move carries it
    from the jet axis to the site.  The sound field only has to supply the small
    difference between that motion and free flight.  Then the funnel takes over.
    """
    drop = pl.drop
    _, x, v, _ = jet.nominal(drop)
    x_a, v_a = x[0], v[0, 2]
    x_pre = site.xs + site.touchdown * drop.a + site.n * gap0
    dz = x_pre[2] - x_a[2]
    T = 2 * dz / v_a                               # uniform deceleration to rest at x_pre
    az = -v_a ** 2 / (2 * dz)
    L = np.linalg.norm(x_pre[:2] - x_a[:2])
    dirv = (x_pre[:2] - x_a[:2]) / L if L > 1e-9 else np.zeros(2)
    n = max(8, int(np.ceil(np.hypot(dz, L) / pl.ds)) + 1)
    # keyframes equally spaced in height
    zs = np.linspace(0, dz, n)
    ts = (v_a - np.sqrt(np.maximum(v_a ** 2 + 2 * az * zs, 0.0))) / (-az)
    u = ts / T
    s_lat = L * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)
    v_lat = L * (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / T
    a_lat = L * (60 * u - 180 * u ** 2 + 120 * u ** 3) / T ** 2
    X = np.stack([x_a[0] + dirv[0] * s_lat, x_a[1] + dirv[1] * s_lat, x_a[2] + zs], 1)
    V = np.stack([dirv[0] * v_lat, dirv[1] * v_lat, v_a + az * ts], 1)
    A = np.stack([dirv[0] * a_lat, dirv[1] * a_lat, np.full(n, az)], 1)
    k_fly = pl.follow(ts, X, V, A)
    k_hold = pl.hold(x_pre, hold_s, k_fly.phi[-1], k_fly.amp[-1])
    k_pr = pl.press(site.xs, site.n, gap0, push=push, phi0=k_hold.phi[-1], n_surf=site.n_surf,
                    gaps=tuple(np.linspace(0.03e-3, gap0, max(5, int(round(gap0 / 0.3e-3)) + 1))))
    keys = concat(concat(k_fly, k_hold), k_pr)
    fun = k_pr.info[0]
    kap = np.array([i["kappa"] for i in k_fly.info])
    clear = float(part.dist(X[: int(0.8 * n)]).min()) if part is not None else np.inf
    summary = dict(mode="direct", t_acq=0.0, t_transit=float(T), t_press=k_pr.t[-1] - k_pr.t[0],
                   t_total=keys.t[-1] - keys.t[0], path_mm=float(np.hypot(dz, L) * 1e3), clearance_mm=clear * 1e3,
                   kappa_min=float(kap.min()), kappa_med=float(np.median(kap[:, 0])),
                   kappa_press_min=float(fun["kappa_t"].min()), funnel_fn_max=float(fun["f_n"].max()),
                   ba_max=float(max(i["Ba"] for i in keys.info)),
                   fres_max=float(max(np.linalg.norm(i["f_res"]) for i in k_fly.info)),
                   lateral_g=float(5.7735 * L / T ** 2 / C.G_ACC), lift_w=float(az / C.G_ACC + 1.0),
                   amp_mean=float(keys.amp.mean()), n_keys=len(keys.t))
    return keys, summary
