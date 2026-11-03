"""
Testing the gas passive scalars advection in RAMSES solver
==========================================================

CI test for the gas passive scalars (``cfg.set_npscal_gas``, experimental). A density top hat
is advected in a uniform periodic flow :math:`v = 1` together with three passive scalars:

* :math:`s_0 = 0.5` everywhere: the scalar fluxes are upwinded on the gas mass flux, so a
  uniform scalar must stay exactly uniform while the density moves through it.
* :math:`s_1`: a top hat in :math:`x`, offset from the density one.
* :math:`s_2 = 1 - s_1`: every step is linear in :math:`s` and the slope limiters are odd,
  so :math:`s_1 + s_2 = 1` must hold exactly. This catches any mix-up between the scalars
  of a cell (they are stored cell-major, :math:`n_{\\rm pscal}` values per cell).

The grid is split into several patches so that the ghost exchange of the scalars between
patches (and MPI ranks) is used. The checks are:

* :math:`s_0` uniform and :math:`s_1 + s_2 = 1` to round-off,
* conservation of :math:`\\sum \\rho s_1` (periodic box),
* :math:`0 \\leq s_1 \\leq 1`,
* the centre of mass of :math:`\\rho s_1` moved by :math:`v t`, and :math:`s_1` is close to the
  exact top hat (L2 error).
"""

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# passive scalars are experimental
shamrock.enable_experimental_features()

npscal = 3
tmax = 0.2

sz = 1 << 1
base = 16

# uniform pressure
press = 1.0
gamma = 1.4


def x_center(rmin, rmax):
    return 0.5 * (rmin[0] + rmax[0])


def rho_map(rmin, rmax):
    return 2.0 if 0.4 < x_center(rmin, rmax) < 0.6 else 1.0


def s1_map(rmin, rmax):
    return 1.0 if 0.2 < x_center(rmin, rmax) < 0.4 else 0.0


def s_maps(rmin, rmax):
    s1 = s1_map(rmin, rmax)
    return (0.5, s1, 1.0 - s1)


def periodic_center(x, m):
    """Centre of mass on the periodic unit interval (the profiles can wrap around)"""
    theta = 2 * np.pi * x
    return np.arctan2(np.sum(m * np.sin(theta)), np.sum(m * np.cos(theta))) / (2 * np.pi)


def run(vx):
    """Advect the top hats with the uniform velocity vx and return the error metrics"""

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    cfg.set_scale_factor(1 / (sz * base))
    cfg.set_eos_gamma(gamma)
    cfg.set_npscal_gas(npscal)
    model.set_solver_config(cfg)

    # small patches, so that the scalars go through the ghost exchange between patches
    model.init_scheduler(1000, 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64(
        "rhoetot", lambda a, b: press / (gamma - 1) + 0.5 * rho_map(a, b) * vx**2
    )
    model.set_field_value_lambda_f64_3("rhovel", lambda a, b: (rho_map(a, b) * vx, 0.0, 0.0))
    for k in range(npscal):
        model.set_field_value_lambda_f64(
            "rho_gas_pscal", lambda a, b, k=k: rho_map(a, b) * s_maps(a, b)[k], k
        )

    def get_state():
        # collective, but the data is only gathered on rank 0
        dic = ctx.collect_data()
        if shamrock.sys.world_rank() != 0:
            return None
        cmin, cmax = dic["cell_min"], dic["cell_max"]
        xc = []
        for i in range(len(cmin)):
            for j in range(8):
                a, b = model.get_cell_coords((tuple(cmin[i]), tuple(cmax[i])), j)
                xc.append(0.5 * (a[0] + b[0]))
        rho = np.array(dic["rho"])
        rho_s = np.array(dic["rho_gas_pscal"]).reshape(-1, npscal)  # cell-major layout
        return np.array(xc), rho, rho_s

    state0 = get_state()

    model.evolve_until(tmax)
    t = model.get_time()

    state = get_state()

    if shamrock.sys.world_rank() != 0:
        return None

    x0, rho0, rho_s0 = state0
    mass_s1_0 = np.sum(rho_s0[:, 1])
    xcm_0 = periodic_center(x0, rho_s0[:, 1])

    x, rho, rho_s = state
    s = rho_s / rho[:, None]

    mass_s1 = np.sum(rho_s[:, 1])
    xcm = periodic_center(x, rho_s[:, 1])
    shift = (xcm - xcm_0 + 0.5) % 1.0 - 0.5

    s1_exact = np.array([1.0 if 0.2 < (xx - vx * t) % 1.0 < 0.4 else 0.0 for xx in x])

    return {
        "max |s0 - 0.5|": np.max(np.abs(s[:, 0] - 0.5)),
        "max |s1 + s2 - 1|": np.max(np.abs(s[:, 1] + s[:, 2] - 1.0)),
        "rel. err. sum rho s1": abs(mass_s1 - mass_s1_0) / mass_s1_0,
        "min s1": np.min(s[:, 1]),
        "max s1": np.max(s[:, 1]),
        "|shift - v t|": abs(shift - vx * t),
        "L2(s1 - exact)": np.sqrt(np.mean((s[:, 1] - s1_exact) ** 2)),
    }


results = {vx: run(vx) for vx in (1.0, -1.0)}


def check(results):
    """Print the metrics and fail on bounds violations (rank 0 only)"""

    for vx, res in results.items():
        print(f"vx = {vx}")
        for name, val in res.items():
            print(f"    {name:<22} = {val}")

    # Round-off level checks, then loose physical bounds: a scalar that is not transported gives
    # a shift of v t = 0.2 and an L2 error of sqrt(0.4) ~ 0.63, while these runs give ~5e-4 and
    # ~0.133
    roundoff = 1e-13
    max_shift_err = 5e-3
    max_l2_s1 = 0.16

    err_log = ""
    for vx, res in results.items():
        checks = [
            ("max |s0 - 0.5|", res["max |s0 - 0.5|"] <= roundoff),
            ("max |s1 + s2 - 1|", res["max |s1 + s2 - 1|"] <= roundoff),
            ("rel. err. sum rho s1", res["rel. err. sum rho s1"] <= roundoff),
            ("min s1", res["min s1"] >= -roundoff),
            ("max s1", res["max s1"] <= 1.0 + roundoff),
            ("|shift - v t|", res["|shift - v t|"] <= max_shift_err),
            ("L2(s1 - exact)", res["L2(s1 - exact)"] <= max_l2_s1),
        ]
        for name, ok in checks:
            if not ok:
                err_log += f"vx = {vx}: {name} = {res[name]} is out of bounds\n"

    if err_log != "":
        exit("Passive scalars advection test failed:\n" + err_log)

    print("ramses_passive_scalars_advection: OK")


# the metrics are only gathered on rank 0
if shamrock.sys.world_rank() == 0:
    check(results)
