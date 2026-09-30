"""Regenerate the photo-z convention figures the tests README shows.

Evaluates the frozen cosmic-shear fiducial under the four runtime
photo-z settings (cspline/Z_LOW default, linear, Steffen, Z_MID; see
test_photoz_conventions.py for what they mean) and plots the
fractional data-vector differences against the default,

    delta xi/xi = xi(setting)/xi(default) - 1,

per tomographic pair and per angular bin, for xi_plus (solid) and
xi_minus (dashed). Masked angular bins are left out. Two figures,
because the two knobs live on different scales:

    photoz_zmid_dxi.png   - the Z_LOW vs Z_MID reading of the n(z)
                            file z column (percent level),
    photoz_interp_dxi.png - linear and Steffen vs cubic spline
                            (1e-4 level).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python ./projects/desy1xplanck/tests/generate_photoz_convention_figure.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import sys
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

EXAMPLE = "example1"
NTOMO = 4
NTHETA = 30
THETA_MIN, THETA_MAX = 0.25, 250.0
MASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "frozen", "data",
                         "DESY3xPlanckR4_6x2pt_Maglim_baseline_archive.mask")

SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))


def datavectors():
    """The printed theory vector under each setting, keyed by tag."""
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({EXAMPLE}, NLA, {tag}) ...", flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            block = info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def plot(curves, fname, title, scale=100.0, unit="%", ylim=None):
    """One 3x5 per-pair panel grid in the notebook-plotter layout.

    curves = {label: (dxi_plus, dxi_minus)}, each a (npair, NTHETA)
    fractional-difference array with NaN at masked bins.
    """
    theta = np.geomspace(THETA_MIN, THETA_MAX, NTHETA + 1)
    theta = (2.0 / 3.0) * (theta[1:]**3 - theta[:-1]**3) \
                        / (theta[1:]**2 - theta[:-1]**2)
    pairs = [(i, j) for i in range(NTOMO) for j in range(i, NTOMO)]
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 7),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        for q, (label, (dp, dm)) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            ax.semilogx(theta, scale * dp[p], color=color, lw=1.6,
                        label=rf"$\xi_+$: {label}" if p == 0 else None)
            ax.semilogx(theta, scale * dm[p], color=color, lw=1.6, ls="--",
                        label=rf"$\xi_-$: {label}" if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$", transform=ax.transAxes,
                fontsize=15)
        if p >= 5:
            ax.set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
        if p % 5 == 0:
            ax.set_ylabel(rf"$\Delta\xi_\pm/\xi_\pm$ [{unit}]", fontsize=16)
    if ylim is not None:
        # fractional differences blow up where xi_plus crosses zero at
        # the largest angles; a fixed range keeps the flat offsets,
        # which are the physics, readable (the spikes run off-panel)
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


def main():
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = datavectors()

    mask = np.loadtxt(MASK_FILE)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = NTOMO * (NTOMO + 1) // 2
    nxi = npair * NTHETA

    def frac(tag):
        # first the xi_plus block, then xi_minus; fractional difference
        # against the default, NaN where the mask removes the point
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        out = []
        for s in range(2):
            sl = slice(s * nxi, (s + 1) * nxi)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(mask[sl] > 0, cur[sl] / ref[sl] - 1.0, np.nan)
            out.append(d.reshape(npair, NTHETA))
        return out

    plot({"Z_MID": frac("Z_MID")},
         "photoz_zmid_dxi.png",
         "desy1xplanck cosmic shear: n(z) z-column read as Z_MID instead of "
         "Z_LOW (frozen fiducial)", scale=100.0, unit="%", ylim=4.0)
    plot({"linear": frac("linear"), "steffen": frac("steffen")},
         "photoz_interp_dxi.png",
         "desy1xplanck cosmic shear: linear and Steffen n(z) interpolation "
         "vs cubic spline (frozen fiducial)", scale=1.0e4,
         unit=r"$10^{-4}$", ylim=8.0)


if __name__ == "__main__":
    main()
