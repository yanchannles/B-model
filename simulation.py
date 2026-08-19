#!/usr/bin/env python3
"""Keller et al. (2013) homogeneous melt-diapir setup with a pseudo-transient solver.

This copy keeps the grid layout and non-linear plastic/marker logic of the
reference MATLAB-style script, while changing the default model setup to the
Keller et al. (2013) Section 3.1 / d118r2 homogeneous host-rock run:
6 km by 4 km rock box plus a 0.5 km sticky-air layer, a 20 percent Gaussian
melt pulse on the lower rock boundary, constant side-boundary extension,
gravity-driven melt segregation, a constant lithostatic fluid-pressure source
below the pulse, and Keller's 50 MPa pressure offset.  The full 4.5 km
computational domain, including sticky air, is plotted.

Grid layout follows the MATLAB code:

  Basic nodes, shape (Ny, Nx):
      ETA, ETA0, GGG, SXY, SXY0, COH, TEN, FRI, YNY
      x = 0:dx:xsize, y = 0:dy:ysize

  Vx nodes, shape (Ny+1, Nx+1):
      vx, qxD, PHIX, KXOE = kphi/etafluid
      xvx = 0:dx:xsize+dx, yvx = -dy/2:dy:ysize+dy/2

  Vy nodes, shape (Ny+1, Nx+1):
      vy, qyD, PHIY, KYOE = kphi/etafluid
      xvy = -dx/2:dx:xsize+dx/2, yvy = 0:dy:ysize+dy

  P nodes, shape (Ny+1, Nx+1):
      pr, pf, PHI, BETTAPHI, ETAP, XI0, XI, YNYT, SXX, SYY, SXX0, SYY0, GGGP
      xp = -dx/2:dx:xsize+dx/2, yp = -dy/2:dy:ysize+dy/2

The active-set plastic correction is written in MATLAB terms: ETA5/YNY5, DSY,
ynpl, ddd, and YERRNOD.  Marker-to-node and node-to-marker interpolation uses
the same four-node bilinear rules and the same yielded-node harmonic update for
etavpm for shear and xivpm for compaction.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

PI = math.pi
SECONDS_PER_KYR = 1000.0 * 365.25 * 24.0 * 3600.0


@dataclass
class Config:
    xsize: float = 6_000.0
    # Total computational height, including sticky air.  The default keeps
    # the original 4.0 km rock column and adds 0.5 km of sticky air.
    ysize: float = 4_500.0
    nx: int = 181
    ny: int = 136
    nt: int = 1500
    niter: int = 8000
    nout: int = 50
    nsave: int = 5
    epsi: float = 1.0e-7

    # Keller et al. (2013), Section 3.1 / d118r2-like physical setup.
    strainrate: float = 1.0e-15
    weak_layer_thickness: float = 0.0
    inclusion_halfwidth: float = 0.0
    sticky_air_thickness: float = 500.0
    eta_air: float = 1.0e16
    rho_air: float = 1.0
    phi_background: float = 0.0
    phi_amplitude: float = 0.20
    gaussian_sigma_x: float = 300.0
    gaussian_sigma_y: float = 300.0
    source_halfwidth: float = 650.0
    # Retained for backward-compatible command-line parsing.  In the pure Pf
    # boundary-condition version, the lower-source flux is not clipped by this
    # parameter; qyD is computed from the prescribed Pf ghost-row boundary.
    source_flux_limit: float = 5.0e-10
    surface_pressure: float = 50.0e6
    # Keller Appendix A.4 material-law regularization.  Physical/state phi
    # remains in [0, 1], while property coefficients use independent minima:
    # phi_f,mat=max(phi,1e-6), phi_s,mat=max(1-phi,1e-2).
    phimin: float = 1.0e-6
    solid_fraction_min: float = 1.0e-2
    # Keller Appendix A4 under-connected threshold. Below this value the
    # pressure system is projected to the one-pressure limit Pf=P, while the
    # permeability remains at the non-zero stabilization cutoff kphi_min.
    phi_crit: float = 1.0e-3
    # Treat only values indistinguishable from true phi=1 as full melt. The
    # 1% solid-fraction cutoff belongs to material laws, not to the state bound.
    full_melt_eps: float = 1.0e-9
    G0: float = 5.0e10
    Ks: float = 1.0e11
    # Keller-style pore modulus: K_phi = Kphi0 * phi**(-Kphi_exp).
    # BETTAPHI stores 1/K_phi = phi**Kphi_exp / Kphi0.
    # This is independent of the shear modulus G0.
    Kphi0: float = 5.0e9
    Kphi_exp: float = 0.5
    eta_block: float = 1.0e18
    eta_weak: float = 1.0e18
    eta_min: float = 1.0e16
    # Keller full-melt-limit regularization: when the mixture becomes
    # fluid-dominated, use this lower cut-off viscosity in the total
    # deviatoric stress, following the high-melt approximation of eq. (63).
    eta_melt_cutoff: float = 1.0e16
    eta_max: float = 1.0e23
    alphaphi: float = 27.0
    coh0: float = 4.0e7
    tens0: float = 2.0e7
    fric_block: float = 0.5
    # Keller permeability prefactor k0 in k_phi = k0*phi**3*(1-phi)**2.
    kphi0: float = 1.0e-8
    # Keller Appendix A4: use a non-zero lower permeability cut-off for Q1Q1 stability.
    kphi_min: float = 1.0e-19
    etafluid: float = 10.0
    rho_solid: float = 3000.0
    rho_fluid: float = 2500.0
    gravity: float = 10.0

    # Pseudo-transient controls retained from the compact solver.
    # Numerical reference melt fraction used only to scale the legacy P-T
    # fluid-pressure/Darcy relaxation. It is not a physical background phi.
    phi_pt_scale: float = 0.01
    CFL: float = 1.0 / 7.1
    Re: float = 3.0 * math.sqrt(10.0) / 2.0 * PI

    # MATLAB timestep/plastic/marker controls.
    dt0: float = 1.0e10
    dtkoef: float = 1.2
    dtkoefup: float = 1.2
    dtstep: int = 200
    nplast: int = 200
    yerrmax: float = 3.0e5
    etawt: float = 0.0
    dxymax: float = 0.3
    dphimax: float = 0.1
    vpratio: float = 1.0 / 3.0
    markers_per_cell: int = 4
    marker_seed: int = 1
    marker_jitter: float = 1.0
    # Keller Appendix-A.5-style marker reseeding.  The target number per cell
    # is markers_per_cell**2.  A cell is rebuilt when its marker count differs
    # from that target by more than marker_reseed_tolerance (25% by default).
    marker_reseed: bool = True
    marker_reseed_tolerance: float = 0.25
    # Numerical reservoir helper.  Default OFF: Keller's lower source is a
    # boundary pressure/flux condition, not an internal porosity/viscosity floor.
    # Enabling this can create a persistent low-viscosity high-porosity tail.
    keep_source_porosity: bool = False
    max_dt_retries: int = 20

    @property
    def nx1(self) -> int:
        return self.nx + 1

    @property
    def ny1(self) -> int:
        return self.ny + 1

    @property
    def dx(self) -> float:
        return self.xsize / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.ysize / (self.ny - 1)

    @property
    def phimax(self) -> float:
        """Upper bound of the physical/state melt fraction."""
        return 1.0

    @property
    def source_x(self) -> float:
        return self.xsize / 2.0

    @property
    def rock_thickness(self) -> float:
        """Thickness of the rock column below the sticky-air layer."""
        return max(self.ysize - self.sticky_air_thickness, 0.0)

    @property
    def rock_top(self) -> float:
        """Initial air-rock interface depth measured from the domain top."""
        return self.sticky_air_thickness

    @property
    def source_column_mean_phi(self) -> float:
        """Initial source-bar mean melt fraction through the rock column."""
        horizontal_mean = (
            self.gaussian_sigma_x
            * math.sqrt(math.pi / 2.0)
            * math.erf(
                self.source_halfwidth
                / (math.sqrt(2.0) * self.gaussian_sigma_x)
            )
            / self.source_halfwidth
        )
        vertical_mean = (
            self.gaussian_sigma_y
            * math.sqrt(math.pi / 2.0)
            * math.erf(
                self.rock_thickness
                / (math.sqrt(2.0) * self.gaussian_sigma_y)
            )
            / max(self.rock_thickness, 1.0e-300)
        )
        return (
            self.phi_background
            + self.phi_amplitude * horizontal_mean * vertical_mean
        )

    @property
    def source_density(self) -> float:
        """Density used to prescribe the lower lithostatic fluid pressure."""
        # Keller's lower reservoir is referenced to the lithostatic pressure
        # of the solid host rock, not to the source-column mean mixture density.
        return self.rho_solid

    @property
    def source_pressure(self) -> float:
        """Constant lithostatic Pf prescribed at the physical bottom face."""
        # Include the negligible sticky-air overburden, then integrate the
        # pure host-rock density through the 4 km rock column.  With the
        # defaults this is 170.005 MPa, effectively Keller's 170 MPa datum.
        return self.surface_pressure + self.gravity * (
            self.rho_air * self.sticky_air_thickness
            + self.rho_solid * self.rock_thickness
        )

    @property
    def phi_full_crit(self) -> float:
        # Only the true full-melt endpoint is projected to the single-fluid
        # limit. Do not confuse the 1% solid-fraction material cutoff with physical phi=1.
        return 1.0 - self.full_melt_eps

    @property
    def phi_dry_crit(self) -> float:
        # At exact phi=0 (and, when enabled, under-connected phi<phi_crit),
        # Pc vanishes but permeability retains the Appendix-A4 cutoff.
        return self.phi_crit

    @property
    def phi_fluid_background_mat(self) -> float:
        """A.4-regularized melt fraction for the zero-melt background."""
        return max(self.phi_background, self.phimin)

    @property
    def phi_solid_background_mat(self) -> float:
        """A.4-regularized solid fraction for the zero-melt background."""
        return max(1.0 - self.phi_background, self.solid_fraction_min)

    @property
    def eta_background(self) -> float:
        """Background shear viscosity evaluated with the A.4 melt cutoff."""
        eta = self.eta_block * math.exp(-self.alphaphi * self.phi_fluid_background_mat)
        return max(eta, self.eta_melt_cutoff)

    @property
    def beta_phi_background(self) -> float:
        """Background pore compressibility evaluated with the A.4 melt cutoff."""
        return self.phi_fluid_background_mat ** self.Kphi_exp / self.Kphi0

    @property
    def kphi_background(self) -> float:
        """Background permeability evaluated with A.4 fractions and k cutoff."""
        kphi = (
            self.kphi0
            * self.phi_fluid_background_mat**3
            * self.phi_solid_background_mat**2
        )
        return max(kphi, self.kphi_min)

    @property
    def xi_background(self) -> float:
        """Background compaction viscosity evaluated with the A.4 melt cutoff."""
        return self.eta_block / self.phi_fluid_background_mat

    @property
    def kphi_pt_scale(self) -> float:
        """Permeability used only to scale the legacy P-T relaxation."""
        return (
            self.kphi0
            * self.phi_pt_scale**3
            * (1.0 - self.phi_pt_scale) ** 2
        )

    @property
    def xi_pt_scale(self) -> float:
        """Compaction viscosity used only to scale the legacy P-T relaxation."""
        return self.eta_block / self.phi_pt_scale

    @property
    def xi_min(self) -> float:
        # At phi=1 the regularized fluid fraction is still exactly one.
        return self.eta_min

    @property
    def xi_max(self) -> float:
        # Upper bound consistent with xi = eta/phi.
        return self.eta_max / self.phimin


@dataclass
class State:
    # Basic nodes (Ny, Nx)
    ETA: np.ndarray
    ETA0: np.ndarray
    GGG: np.ndarray
    EXY: np.ndarray
    SXY: np.ndarray
    SXY0: np.ndarray
    COH: np.ndarray
    TEN: np.ndarray
    FRI: np.ndarray
    YNY: np.ndarray
    SIIB: np.ndarray
    SYIELD: np.ndarray
    DSY: np.ndarray
    wyx: np.ndarray

    # Vx nodes (Ny+1, Nx+1)
    vx: np.ndarray
    qxD: np.ndarray
    PHIX: np.ndarray
    KXOE: np.ndarray

    # Vy nodes (Ny+1, Nx+1)
    vy: np.ndarray
    qyD: np.ndarray
    PHIY: np.ndarray
    KYOE: np.ndarray

    # P nodes (Ny+1, Nx+1)
    RHO: np.ndarray
    pr: np.ndarray
    pf: np.ndarray
    pr0: np.ndarray
    pf0: np.ndarray
    PHI: np.ndarray
    PHI0: np.ndarray
    BETTAPHI: np.ndarray
    ETAP: np.ndarray
    XI: np.ndarray
    XI0: np.ndarray
    YNYT: np.ndarray
    DSYT: np.ndarray
    GGGP: np.ndarray
    EXX: np.ndarray
    EYY: np.ndarray
    SXX: np.ndarray
    SXX0: np.ndarray
    SYY: np.ndarray
    SYY0: np.ndarray
    DIVV: np.ndarray
    APHI: np.ndarray
    dphidt: np.ndarray
    pf_prev_iter: np.ndarray

    # Step-1 Keller diagnostics only; not used by the solver.
    # Pc = pr - pf is the compaction pressure implied by the code variables.
    # solidP/solidB = max(1 - phi, phimin) on P/basic nodes.
    Pc: np.ndarray
    solidP: np.ndarray
    solidB: np.ndarray

    # Nodal velocities at pressure nodes for marker advection.
    vxp: np.ndarray
    vyp: np.ndarray

    # Work arrays for convergence.
    vydif: np.ndarray
    vy_prev_iter: np.ndarray

    # Markers.
    xm: np.ndarray
    ym: np.ndarray
    tm: np.ndarray
    phim: np.ndarray
    etavpm: np.ndarray
    xivpm: np.ndarray
    sxxm: np.ndarray
    syym: np.ndarray
    sxym: np.ndarray


@dataclass
class PlasticResult:
    ETA5: np.ndarray
    YNY5: np.ndarray
    yerr: float
    ynpl: int
    yny_old_count: int
    yny5_count: int
    yny_changed_count: int


@dataclass
class TensilePlasticResult:
    XI5: np.ndarray
    YNYT5: np.ndarray
    yerr: float
    ynpl: int
    ynyt_old_count: int
    ynyt5_count: int
    ynyt_changed_count: int
    invalid_count: int


@dataclass
class StepStats:
    pt_iters: int
    resid: float
    err_vy: float
    dt: float
    dtm: float
    iplast: int
    yerr: float
    ynpl: int
    yny_count: int
    yny5_count: int
    yny_changed_count: int
    tyerr: float
    tynpl: int
    tyny_count: int
    tyny5_count: int
    tyny_changed_count: int
    tinvalid_count: int
    retries: int
    converged: bool
    pt_converged: bool


def finite_min(a: np.ndarray) -> float:
    vals = np.asarray(a)[np.isfinite(a)]
    return float(np.min(vals)) if vals.size else float("nan")


def finite_max(a: np.ndarray) -> float:
    vals = np.asarray(a)[np.isfinite(a)]
    return float(np.max(vals)) if vals.size else float("nan")


def finite_mean(a: np.ndarray) -> float:
    vals = np.asarray(a)[np.isfinite(a)]
    return float(np.mean(vals)) if vals.size else float("nan")


def finite_abs_max(a: np.ndarray) -> float:
    vals = np.asarray(a)[np.isfinite(a)]
    return float(np.max(np.abs(vals))) if vals.size else float("nan")


class RuntimeLogger:
    """Detailed runtime logger for P-T and plastic iterations.

    CSV files contain every logged iteration and are never overwritten.  Console
    output can be throttled independently so long runs remain readable.
    """

    pt_fields = [
        "step", "iplast", "itpt", "dt", "dt_rho", "Gdt", "Kdt",
        "resid", "err_v", "err_vy", "err_pf_diff", "err_pf",
        "vx_abs_max", "vy_abs_max", "pr_abs_max", "pf_abs_max", "pc_abs_max",
        "divv_abs_max", "dphidt_abs_max", "qx_abs_max", "qy_abs_max",
        "phi_min", "phi_mean", "phi_max", "eta_min", "eta_mean", "eta_max",
        "xi_min", "xi_mean", "xi_max", "converged",
    ]
    plastic_fields = [
        "step", "iplast", "dt", "pt_iters_this_plastic", "pt_resid", "pt_converged",
        "shear_converged", "tensile_converged", "plastic_converged",
        "yerr", "ynpl", "yny_old_count", "yny5_count", "yny_changed_count",
        "tyerr", "tynpl", "ynyt_old_count", "ynyt5_count", "ynyt_changed_count", "tinvalid_count",
        "eta_min", "eta_mean", "eta_max", "eta5_min", "eta5_mean", "eta5_max",
        "eta_log10_delta_max", "eta_nodes_changed",
        "xi_min", "xi_mean", "xi_max", "xi5_min", "xi5_mean", "xi5_max",
        "xi_log10_delta_max", "xi_nodes_changed",
        "siib_max", "syield_min", "syield_mean", "syield_max", "dsy_abs_max",
        "pc_abs_max", "phi_min", "phi_mean", "phi_max", "retries",
    ]

    def __init__(
        self,
        outdir: Path,
        *,
        enabled: bool,
        console: bool,
        pt_log_every: int,
        pt_console_every: int,
        plastic_console_every: int,
    ) -> None:
        self.write_csv = enabled
        self.console = console
        self.enabled = self.write_csv or self.console
        self.pt_log_every = max(1, int(pt_log_every))
        self.pt_console_every = max(0, int(pt_console_every))
        self.plastic_console_every = max(0, int(plastic_console_every))
        self.pt_file = None
        self.plastic_file = None
        self.pt_writer = None
        self.plastic_writer = None
        self.pt_path: Path | None = None
        self.plastic_path: Path | None = None
        if not self.write_csv:
            return

        log_dir = outdir / "runtime_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.pt_path = self._unique_path(log_dir, "pt_iterations", ".csv")
        self.plastic_path = self._unique_path(log_dir, "plastic_iterations", ".csv")
        self.pt_file = self.pt_path.open("w", newline="")
        self.plastic_file = self.plastic_path.open("w", newline="")
        self.pt_writer = csv.DictWriter(self.pt_file, fieldnames=self.pt_fields)
        self.plastic_writer = csv.DictWriter(self.plastic_file, fieldnames=self.plastic_fields)
        self.pt_writer.writeheader()
        self.plastic_writer.writeheader()

    @staticmethod
    def _unique_path(log_dir: Path, stem: str, suffix: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"{stem}_{stamp}{suffix}"
        counter = 1
        while path.exists():
            path = log_dir / f"{stem}_{stamp}_{counter:02d}{suffix}"
            counter += 1
        return path

    def close(self) -> None:
        for f in (self.pt_file, self.plastic_file):
            if f is not None:
                f.flush()
                f.close()

    def should_log_pt(self, itpt_one_based: int) -> bool:
        return self.write_csv and itpt_one_based % self.pt_log_every == 0

    def should_print_final_pt(self) -> bool:
        return self.console and self.pt_console_every > 0

    def log_pt(
        self,
        st: State,
        cfg: Config,
        *,
        step: int,
        iplast: int,
        itpt: int,
        dt: float,
        dt_rho: float,
        Gdt: float,
        Kdt: float,
        resid: float,
        err_v: float,
        err_vy: float,
        err_pf_diff: float,
        err_pf: float,
        converged: bool,
        write_csv: bool = True,
        print_console: bool = False,
    ) -> None:
        if not self.enabled:
            return
        ip = slice(1, cfg.ny)
        jp = slice(1, cfg.nx)
        phi = st.PHI[ip, jp]
        row = {
            "step": step + 1,
            "iplast": iplast,
            "itpt": itpt,
            "dt": dt,
            "dt_rho": dt_rho,
            "Gdt": Gdt,
            "Kdt": Kdt,
            "resid": resid,
            "err_v": err_v,
            "err_vy": err_vy,
            "err_pf_diff": err_pf_diff,
            "err_pf": err_pf,
            "vx_abs_max": finite_abs_max(st.vx),
            "vy_abs_max": finite_abs_max(st.vy),
            "pr_abs_max": finite_abs_max(st.pr[ip, jp]),
            "pf_abs_max": finite_abs_max(st.pf[ip, jp]),
            "pc_abs_max": finite_abs_max(st.pr[ip, jp] - st.pf[ip, jp]),
            "divv_abs_max": finite_abs_max(st.DIVV[ip, jp]),
            "dphidt_abs_max": finite_abs_max(st.dphidt[ip, jp]),
            "qx_abs_max": finite_abs_max(st.qxD),
            "qy_abs_max": finite_abs_max(st.qyD),
            "phi_min": finite_min(phi),
            "phi_mean": finite_mean(phi),
            "phi_max": finite_max(phi),
            "eta_min": finite_min(st.ETA),
            "eta_mean": finite_mean(st.ETA),
            "eta_max": finite_max(st.ETA),
            "xi_min": finite_min(st.XI[ip, jp]),
            "xi_mean": finite_mean(st.XI[ip, jp]),
            "xi_max": finite_max(st.XI[ip, jp]),
            "converged": converged,
        }
        if write_csv and self.pt_writer is not None:
            self.pt_writer.writerow(row)
        if write_csv and self.pt_file is not None:
            self.pt_file.flush()
        if self.console and print_console:
            print(
                f"[PT] step={step + 1:04d} iplast={iplast:04d} itpt={itpt:05d} "
                f"resid={resid:.3e} dVy={err_v:.3e} dPf={err_pf_diff:.3e} "
                f"|Vy|max={err_vy:.3e} |Pc|max={row['pc_abs_max']:.3e} "
                f"|q|max={max(row['qx_abs_max'], row['qy_abs_max']):.3e}",
                flush=True,
            )

    def log_plastic(
        self,
        st: State,
        cfg: Config,
        pr: PlasticResult,
        tr: TensilePlasticResult,
        *,
        step: int,
        iplast: int,
        dt: float,
        pt_iters_this_plastic: int,
        pt_resid: float,
        pt_converged: bool,
        shear_converged: bool,
        tensile_converged: bool,
        plastic_converged: bool,
        retries: int,
    ) -> None:
        if not self.enabled:
            return
        ip = slice(1, cfg.ny)
        jp = slice(1, cfg.nx)
        phi = st.PHI[ip, jp]
        eta_log_delta = np.abs(np.log10(np.maximum(pr.ETA5, 1.0e-300)) - np.log10(np.maximum(st.ETA, 1.0e-300)))
        xi_log_delta = np.abs(np.log10(np.maximum(tr.XI5[ip, jp], 1.0e-300)) - np.log10(np.maximum(st.XI[ip, jp], 1.0e-300)))
        finite_syield = st.SYIELD[np.isfinite(st.SYIELD)]
        row = {
            "step": step + 1,
            "iplast": iplast,
            "dt": dt,
            "pt_iters_this_plastic": pt_iters_this_plastic,
            "pt_resid": pt_resid,
            "pt_converged": pt_converged,
            "shear_converged": shear_converged,
            "tensile_converged": tensile_converged,
            "plastic_converged": plastic_converged,
            "yerr": pr.yerr,
            "ynpl": pr.ynpl,
            "yny_old_count": pr.yny_old_count,
            "yny5_count": pr.yny5_count,
            "yny_changed_count": pr.yny_changed_count,
            "tyerr": tr.yerr,
            "tynpl": tr.ynpl,
            "ynyt_old_count": tr.ynyt_old_count,
            "ynyt5_count": tr.ynyt5_count,
            "ynyt_changed_count": tr.ynyt_changed_count,
            "tinvalid_count": tr.invalid_count,
            "eta_min": finite_min(st.ETA),
            "eta_mean": finite_mean(st.ETA),
            "eta_max": finite_max(st.ETA),
            "eta5_min": finite_min(pr.ETA5),
            "eta5_mean": finite_mean(pr.ETA5),
            "eta5_max": finite_max(pr.ETA5),
            "eta_log10_delta_max": finite_abs_max(eta_log_delta),
            "eta_nodes_changed": int(np.count_nonzero(eta_log_delta > 1.0e-12)),
            "xi_min": finite_min(st.XI[ip, jp]),
            "xi_mean": finite_mean(st.XI[ip, jp]),
            "xi_max": finite_max(st.XI[ip, jp]),
            "xi5_min": finite_min(tr.XI5[ip, jp]),
            "xi5_mean": finite_mean(tr.XI5[ip, jp]),
            "xi5_max": finite_max(tr.XI5[ip, jp]),
            "xi_log10_delta_max": finite_abs_max(xi_log_delta),
            "xi_nodes_changed": int(np.count_nonzero(xi_log_delta > 1.0e-12)),
            "siib_max": finite_max(st.SIIB),
            "syield_min": finite_min(finite_syield),
            "syield_mean": finite_mean(finite_syield),
            "syield_max": finite_max(finite_syield),
            "dsy_abs_max": finite_abs_max(st.DSY),
            "pc_abs_max": finite_abs_max(st.pr[ip, jp] - st.pf[ip, jp]),
            "phi_min": finite_min(phi),
            "phi_mean": finite_mean(phi),
            "phi_max": finite_max(phi),
            "retries": retries,
        }
        if self.plastic_writer is not None:
            self.plastic_writer.writerow(row)
        if self.plastic_file is not None:
            self.plastic_file.flush()
        if self.console and self.plastic_console_every and iplast % self.plastic_console_every == 0:
            print(
                f"[PL] step={step + 1:04d} iplast={iplast:04d} pt_iter={pt_iters_this_plastic:05d} "
                f"resid={pt_resid:.3e} YERR={pr.yerr:.3e} ynpl={pr.ynpl} "
                f"YNY(old->new,d)={pr.yny_old_count}->{pr.yny5_count},{pr.yny_changed_count} "
                f"TYERR={tr.yerr:.3e} tynpl={tr.ynpl} "
                f"YNYT(old->new,d)={tr.ynyt_old_count}->{tr.ynyt5_count},{tr.ynyt_changed_count} "
                f"dlogETAmax={row['eta_log10_delta_max']:.3e} dlogXImax={row['xi_log10_delta_max']:.3e} "
                f"|Pc|max={row['pc_abs_max']:.3e} pt_conv={pt_converged} pl_conv={plastic_converged}",
                flush=True,
            )



def validate_config(cfg: Config) -> None:
    """Reject invalid parameter combinations instead of silently modifying them."""
    if not 0.0 <= cfg.phi_background <= 1.0:
        raise ValueError("phi_background must be in [0, 1].")
    if not 0.0 <= cfg.phi_background + cfg.phi_amplitude <= 1.0:
        raise ValueError("phi_background + phi_amplitude must be in [0, 1].")
    if not 0.0 < cfg.phimin < 1.0:
        raise ValueError("phimin must be in (0, 1).")
    if not 0.0 < cfg.phi_pt_scale < 1.0:
        raise ValueError("phi_pt_scale must be in (0, 1).")
    if not 0.0 < cfg.solid_fraction_min < 1.0:
        raise ValueError("solid_fraction_min must be in (0, 1).")
    if not 0.0 <= cfg.phi_crit < 1.0:
        raise ValueError("phi_crit must be in [0, 1).")
    if not 0.0 < cfg.full_melt_eps < 1.0:
        raise ValueError("full_melt_eps must be in (0, 1).")
    if cfg.phi_crit >= cfg.phi_full_crit:
        raise ValueError("phi_crit must be smaller than phi_full_crit.")
    if cfg.Kphi0 <= 0.0:
        raise ValueError("Kphi0 must be positive.")
    if cfg.kphi0 <= 0.0 or cfg.kphi_min <= 0.0:
        raise ValueError("kphi0 and kphi_min must be positive.")
    if cfg.etafluid <= 0.0:
        raise ValueError("etafluid must be positive.")
    if cfg.markers_per_cell < 1:
        raise ValueError("markers_per_cell must be at least 1.")
    if not 0.0 < cfg.marker_reseed_tolerance < 1.0:
        raise ValueError("marker_reseed_tolerance must be in (0, 1).")


def zeros(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def coords(cfg: Config) -> dict[str, np.ndarray]:
    return {
        "x": np.arange(cfg.nx, dtype=np.float64) * cfg.dx,
        "y": np.arange(cfg.ny, dtype=np.float64) * cfg.dy,
        "xvx": np.arange(cfg.nx1, dtype=np.float64) * cfg.dx,
        "yvx": -cfg.dy / 2.0 + np.arange(cfg.ny1, dtype=np.float64) * cfg.dy,
        "xvy": -cfg.dx / 2.0 + np.arange(cfg.nx1, dtype=np.float64) * cfg.dx,
        "yvy": np.arange(cfg.ny1, dtype=np.float64) * cfg.dy,
        "xp": -cfg.dx / 2.0 + np.arange(cfg.nx1, dtype=np.float64) * cfg.dx,
        "yp": -cfg.dy / 2.0 + np.arange(cfg.ny1, dtype=np.float64) * cfg.dy,
    }


def material_phase_fractions(phi: np.ndarray | float, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical phi and independently regularized A.4 phase fractions.

    The physical/state melt fraction remains in [0, 1].  Only coefficients in
    melt/solid-fraction-dependent material laws are regularized:
        phi_f,mat = max(phi, phimin)
        phi_s,mat = max(1-phi, solid_fraction_min)
    The two regularized fractions are numerical coefficients and need not sum
    to one at the pure-phase endpoints.
    """
    phi_state = np.clip(np.asarray(phi, dtype=np.float64), 0.0, 1.0)
    phi_fluid_mat = np.maximum(phi_state, cfg.phimin)
    phi_solid_mat = np.maximum(1.0 - phi_state, cfg.solid_fraction_min)
    return phi_state, phi_fluid_mat, phi_solid_mat


def material_intrinsic_viscosity(tm: np.ndarray, cfg: Config) -> np.ndarray:
    """Marker intrinsic/background viscosity eta0 selected only by material type.

    This is Keller's eta0 in eqs. (50)-(51): it does not include melt
    weakening and does not include plastic weakening.  Material type 1 uses
    eta_block; optional material types 2 and 3 use eta_weak.  The Keller
    default setup generated here is homogeneous, so only material type 1 is
    present unless weak layers/inclusions are explicitly re-enabled.
    """
    out = np.full(tm.shape, cfg.eta_block, dtype=np.float64)
    out[tm == 2] = cfg.eta_weak
    out[tm == 3] = cfg.eta_air
    return out


def material_matrix_viscosity(tm: np.ndarray, phi: np.ndarray, cfg: Config) -> np.ndarray:
    """Unyielded shear viscosity ETA0, matching the reference MATLAB logic.

    For tm=1/2 rocks, Keller eq. (50) is used:
    eta = eta0*exp(-alpha_phi*phi), with alpha_phi=27 in Table 2.
    For tm=3 weak/sticky material, the MATLAB code uses etasolidm(tm)
    directly, without melt weakening.
    """
    _, phi_fluid_mat, _ = material_phase_fractions(phi, cfg)
    eta0 = material_intrinsic_viscosity(tm, cfg)
    out = eta0 * np.exp(-cfg.alphaphi * phi_fluid_mat)
    weak = tm == 3
    out[weak] = eta0[weak]
    # Keller notes that high melt fractions require a lower viscosity cut-off
    # for numerical stability.  Keep the solid-matrix state variables from
    # falling far below the cut-off used in the total-stress full-melt limit.
    out = np.maximum(out, cfg.eta_melt_cutoff)
    return out


def material_xi0_viscosity(tm: np.ndarray, phi: np.ndarray, cfg: Config) -> np.ndarray:
    """Unyielded compaction viscosity xi0 on markers.

    This is the P-node analogue of material_matrix_viscosity(): compute the
    material law on markers first, then interpolate it to the grid.  For
    tm=1/2 rocks, Keller eq. (51) with p=1 is used: xi0 = eta0/phi.
    For tm=3 weak/sticky material, match the MATLAB-style treatment of that
    layer by keeping the viscosity independent of porosity.
    """
    _, phi_fluid_mat, _ = material_phase_fractions(phi, cfg)
    eta0 = material_intrinsic_viscosity(tm, cfg)
    out = eta0 / phi_fluid_mat
    weak = tm == 3
    out[weak] = eta0[weak]
    return out


def marker_permeability_over_eta(tm: np.ndarray, phi: np.ndarray, cfg: Config) -> np.ndarray:
    """Keller eq. (62) permeability divided by fluid viscosity.

    k_phi = k0 * phi**3 * (1 - phi)**2.
    The material-type argument is kept for interface symmetry with the marker
    material-property functions, but permeability is currently porosity-only.
    """
    phi_state, phi_fluid_mat, phi_solid_mat = material_phase_fractions(phi, cfg)
    kphi = cfg.kphi0 * phi_fluid_mat**3 * phi_solid_mat**2
    # Keep the non-zero Appendix-A4 stabilization permeability at low phi.
    kphi = np.maximum(kphi, cfg.kphi_min)
    kphi = np.where(phi_state < cfg.phi_dry_crit, cfg.kphi_min, kphi)
    return kphi / cfg.etafluid


def source_reservoir_phi(x: np.ndarray, y: np.ndarray, cfg: Config) -> np.ndarray:
    """Gaussian porosity floor associated with the lower melt reservoir."""
    return np.clip(
        cfg.phi_background
        + cfg.phi_amplitude
        * np.exp(
            -(
                (x - cfg.source_x) ** 2 / (2.0 * cfg.gaussian_sigma_x**2)
                + (y - cfg.ysize) ** 2 / (2.0 * cfg.gaussian_sigma_y**2)
            )
        ),
        0.0,
        1.0,
    )


def recharge_source_markers(st: State, cfg: Config) -> None:
    """Keep the lower source patch as a melt reservoir without changing its flux BC.

    The melt supply condition is the fixed lower fluid pressure only.  This
    helper only prevents marker advection/compaction from leaving dry
    high-viscosity marker history inside that reservoir patch before the next
    marker-to-node interpolation.
    """
    if not cfg.keep_source_porosity:
        return

    source_zone = (
        (st.tm < 3)
        & (st.ym > cfg.ysize - 2.5 * cfg.gaussian_sigma_y)
        & (np.abs(st.xm - cfg.source_x) < 3.0 * cfg.gaussian_sigma_x)
    )
    if not np.any(source_zone):
        return

    phi_floor = source_reservoir_phi(st.xm[source_zone], st.ym[source_zone], cfg)
    st.phim[source_zone] = np.maximum(st.phim[source_zone], phi_floor)

    # Porosity recharge changes the unyielded material law.  Clamp the carried
    # marker viscosity histories to the refreshed material state so a dry marker
    # entering the source strip cannot make the next nodal ETA/XI look frozen.
    eta_floor = material_matrix_viscosity(st.tm[source_zone], st.phim[source_zone], cfg)
    xi_floor = material_xi0_viscosity(st.tm[source_zone], st.phim[source_zone], cfg)
    st.etavpm[source_zone] = np.minimum(st.etavpm[source_zone], eta_floor)
    st.xivpm[source_zone] = np.minimum(st.xivpm[source_zone], xi_floor)


def apply_source_reservoir_nodes(st: State, cfg: Config) -> None:
    """Keep nodal source properties consistent with the marker reservoir.

    This is not a Darcy-flux boundary condition.  It only prevents marker
    depletion/interpolation gaps from turning the lower pressure source into a
    high-viscosity, low-permeability plug.
    """
    if not cfg.keep_source_porosity:
        return

    c = coords(cfg)
    source_y_min = cfg.ysize - 2.5 * cfg.gaussian_sigma_y
    source_x_halfwidth = 3.0 * cfg.gaussian_sigma_x

    def source_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (y > source_y_min) & (np.abs(x - cfg.source_x) < source_x_halfwidth)

    def k_over_eta_floor(phi: np.ndarray) -> np.ndarray:
        _, phi_fluid_mat, phi_solid_mat = material_phase_fractions(phi, cfg)
        kphi = cfg.kphi0 * phi_fluid_mat**3 * phi_solid_mat**2
        return np.maximum(kphi, cfg.kphi_min) / cfg.etafluid

    # P nodes: porosity/storage/compaction reservoir.
    xp = c["xp"][None, :]
    yp = c["yp"][:, None]
    mask_p = source_mask(xp, yp)
    if np.any(mask_p):
        phi_floor = source_reservoir_phi(xp, yp, cfg)
        st.PHI[mask_p] = np.maximum(st.PHI[mask_p], phi_floor[mask_p])
        _, phi_fluid_mat, _ = material_phase_fractions(phi_floor, cfg)
        xi_cap = cfg.eta_block / phi_fluid_mat
        st.XI0[mask_p] = np.minimum(st.XI0[mask_p], xi_cap[mask_p])
        st.XI[mask_p] = np.minimum(st.XI[mask_p], xi_cap[mask_p])
        update_bettaphi_keller(st, cfg)
        st.YNYT[...] = st.XI < st.XI0
        copy_pnode_edges(st.XI0)
        copy_pnode_edges(st.XI)
        copy_pnode_edges(st.YNYT)

    # Vx/Vy nodes: keep permeability and staggered porosity open in the source.
    xvx = c["xvx"][None, :]
    yvx = c["yvx"][:, None]
    mask_x = source_mask(xvx, yvx)
    if np.any(mask_x):
        phi_floor_x = source_reservoir_phi(xvx, yvx, cfg)
        st.PHIX[mask_x] = np.maximum(st.PHIX[mask_x], phi_floor_x[mask_x])
        st.KXOE[mask_x] = np.maximum(st.KXOE[mask_x], k_over_eta_floor(phi_floor_x)[mask_x])

    xvy = c["xvy"][None, :]
    yvy = c["yvy"][:, None]
    mask_y = source_mask(xvy, yvy)
    if np.any(mask_y):
        phi_floor_y = source_reservoir_phi(xvy, yvy, cfg)
        st.PHIY[mask_y] = np.maximum(st.PHIY[mask_y], phi_floor_y[mask_y])
        st.KYOE[mask_y] = np.maximum(st.KYOE[mask_y], k_over_eta_floor(phi_floor_y)[mask_y])

    # Basic nodes: cap unyielded/current shear viscosity in the reservoir.
    xb = c["x"][None, :]
    yb = c["y"][:, None]
    mask_b = source_mask(xb, yb)
    if np.any(mask_b):
        phi_floor_b = source_reservoir_phi(xb, yb, cfg)
        _, phi_fluid_mat, _ = material_phase_fractions(phi_floor_b, cfg)
        eta_cap = cfg.eta_block * np.exp(-cfg.alphaphi * phi_fluid_mat)
        eta_cap = np.maximum(eta_cap, cfg.eta_melt_cutoff)
        st.ETA0[mask_b] = np.minimum(st.ETA0[mask_b], eta_cap[mask_b])
        st.ETA[mask_b] = np.minimum(st.ETA[mask_b], eta_cap[mask_b])
        st.YNY[...] = st.ETA < st.ETA0


def mixture_density(phi: np.ndarray, cfg: Config) -> np.ndarray:
    """Keller mixture density, rho = (1 - phi) rho_s + phi rho_f."""
    phi = np.clip(phi, 0.0, 1.0)
    return (1.0 - phi) * cfg.rho_solid + phi * cfg.rho_fluid


def marker_density(tm: np.ndarray, phi: np.ndarray, cfg: Config) -> np.ndarray:
    """Marker density with a low-density sticky-air material (tm==3)."""
    rho = mixture_density(phi, cfg)
    rho = np.asarray(rho, dtype=np.float64).copy()
    rho[tm == 3] = cfg.rho_air
    return rho


def keller_eq37_shear_yield(
    cohesion: np.ndarray,
    tensile_strength: np.ndarray,
    friction_sin: np.ndarray,
    effective_pressure: np.ndarray,
) -> np.ndarray:
    """Combined Mohr-Coulomb/Griffith yield envelope from Keller eq. (37).

    The code stores FRI as the pressure coefficient used by the MATLAB
    benchmark.  Here it is interpreted as sin(friction_angle), so the default
    FRI=0.6 corresponds to a friction angle of about 36.9 degrees.  If FRI is
    set to sin(30 deg)=0.5, this function reproduces the paper's 30 degree
    example.
    """
    sin_phi = np.clip(friction_sin, 0.0, 1.0 - 1.0e-12)
    cos_phi = np.sqrt(np.maximum(1.0 - sin_phi * sin_phi, 0.0))

    shear_branch = cohesion * cos_phi + effective_pressure * sin_phi
    tensile_branch = tensile_strength + effective_pressure
    transition_pressure = (cohesion * cos_phi - tensile_strength) / np.maximum(1.0 - sin_phi, 1.0e-12)
    return np.maximum(np.where(effective_pressure > transition_pressure, shear_branch, tensile_branch), 0.0)


def initial_markers(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nxm = (cfg.nx - 1) * cfg.markers_per_cell
    nym = (cfg.ny - 1) * cfg.markers_per_cell
    dxm = cfg.xsize / nxm
    dym = cfg.ysize / nym
    x = dxm / 2.0 + np.arange(nxm, dtype=np.float64) * dxm
    y = dym / 2.0 + np.arange(nym, dtype=np.float64) * dym
    X, Y = np.meshgrid(x, y, indexing="xy")  # shape (Nym, Nxm)
    if cfg.marker_jitter > 0.0:
        rng = np.random.default_rng(cfg.marker_seed)
        X = X + (rng.random(X.shape) - 0.5) * dxm * cfg.marker_jitter
        Y = Y + (rng.random(Y.shape) - 0.5) * dym * cfg.marker_jitter
    X = np.clip(X, 1.0e-9, cfg.xsize - 1.0e-9)
    Y = np.clip(Y, 1.0e-9, cfg.ysize - 1.0e-9)

    xm = X.ravel()
    ym = Y.ravel()
    tm = np.ones_like(xm, dtype=np.int8)
    if cfg.sticky_air_thickness > 0.0:
        tm[ym < cfg.sticky_air_thickness] = 3
    if cfg.weak_layer_thickness > 0.0:
        weak_rock = (ym >= cfg.sticky_air_thickness) & (
            (ym < cfg.sticky_air_thickness + cfg.weak_layer_thickness)
            | (ym > cfg.ysize - cfg.weak_layer_thickness)
        )
        tm[weak_rock] = 3
    if cfg.inclusion_halfwidth > 0.0:
        tm[(np.abs(xm - cfg.xsize / 2.0) < cfg.inclusion_halfwidth) & (np.abs(ym - cfg.ysize / 2.0) < cfg.inclusion_halfwidth)] = 2

    # Keller Fig. 3: a 2-D Gaussian pulse with 20 percent peak melt fraction,
    # centered at the middle of the lower boundary in an otherwise dry host.
    phi_gaussian = cfg.phi_amplitude * np.exp(
        -(
            (xm - cfg.source_x) ** 2 / (2.0 * cfg.gaussian_sigma_x**2)
            + (ym - cfg.ysize) ** 2 / (2.0 * cfg.gaussian_sigma_y**2)
        )
    )
    phim = np.clip(cfg.phi_background + phi_gaussian, 0.0, 1.0)
    # Sticky air is a separate, dry, low-density material rather than melt.
    phim[tm == 3] = 0.0
    etavpm = material_matrix_viscosity(tm, phim, cfg)
    xivpm = material_xi0_viscosity(tm, phim, cfg)
    sxxm = np.zeros_like(xm)
    syym = np.zeros_like(xm)
    sxym = np.zeros_like(xm)
    return xm, ym, tm, phim, etavpm, xivpm, sxxm, syym, sxym


def bilinear_indices_weights(
    x: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, int],
    cfg: Config,
    *,
    x0: float,
    y0: float,
    max_i0: int | None = None,
    max_j0: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = shape
    if max_i0 is None:
        max_i0 = ny - 2
    if max_j0 is None:
        max_j0 = nx - 2
    xi = (x - x0) / cfg.dx
    yi = (y - y0) / cfg.dy
    j = np.floor(xi).astype(np.int64)
    i = np.floor(yi).astype(np.int64)
    j = np.clip(j, 0, max_j0)
    i = np.clip(i, 0, max_i0)
    # fx = np.clip(xi - j, 0.0, 1.0)
    # fy = np.clip(yi - i, 0.0, 1.0)
    fx = xi - j
    fy = yi - i
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy
    return i, j, w00, w10, w01, w11


def scatter_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    cfg: Config,
    *,
    x0: float,
    y0: float,
    max_i0: int | None = None,
    max_j0: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    i, j, w00, w10, w01, w11 = bilinear_indices_weights(x, y, shape, cfg, x0=x0, y0=y0, max_i0=max_i0, max_j0=max_j0)
    acc = np.zeros(shape, dtype=np.float64)
    wsum = np.zeros(shape, dtype=np.float64)
    np.add.at(acc, (i, j), values * w00)
    np.add.at(acc, (i, j + 1), values * w10)
    np.add.at(acc, (i + 1, j), values * w01)
    np.add.at(acc, (i + 1, j + 1), values * w11)
    np.add.at(wsum, (i, j), w00)
    np.add.at(wsum, (i, j + 1), w10)
    np.add.at(wsum, (i + 1, j), w01)
    np.add.at(wsum, (i + 1, j + 1), w11)
    return acc, wsum


def interp_from_grid(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    *,
    x0: float,
    y0: float,
    max_i0: int | None = None,
    max_j0: int | None = None,
) -> np.ndarray:
    i, j, w00, w10, w01, w11 = bilinear_indices_weights(x, y, field.shape, cfg, x0=x0, y0=y0, max_i0=max_i0, max_j0=max_j0)
    return (
        field[i, j] * w00
        + field[i, j + 1] * w10
        + field[i + 1, j] * w01
        + field[i + 1, j + 1] * w11
    )


def average_scatter(acc: np.ndarray, wsum: np.ndarray, default: float | np.ndarray) -> np.ndarray:
    if np.isscalar(default):
        out = np.full(acc.shape, float(default), dtype=np.float64)
    else:
        out = np.array(default, copy=True, dtype=np.float64)
    mask = wsum > 0.0
    out[mask] = acc[mask] / wsum[mask]
    return out


def copy_pnode_edges(a: np.ndarray) -> None:
    # MATLAB symmetry for P-node external rows/columns.
    ny1, nx1 = a.shape
    ny = ny1 - 1
    nx = nx1 - 1
    a[0, 1:nx] = a[1, 1:nx]
    a[ny, 1:nx] = a[ny - 1, 1:nx]
    a[:, 0] = a[:, 1]
    a[:, nx] = a[:, nx - 1]


def update_bettaphi_keller(st: State, cfg: Config) -> None:
    """Update BETTAPHI using Keller's pore modulus law.

    K_phi = Kphi0 * phi**(-Kphi_exp), so BETTAPHI = 1/K_phi.
    This replaces the previous code-specific choice BETTAPHI = phi/G.
    """
    _, phi_fluid_mat, _ = material_phase_fractions(st.PHI, cfg)
    st.BETTAPHI[...] = phi_fluid_mat ** cfg.Kphi_exp / cfg.Kphi0
    copy_pnode_edges(st.BETTAPHI)


def update_keller_step1_diagnostics(st: State, cfg: Config) -> None:
    """Update Keller comparison diagnostics without feeding them back into the solver.

    Pc is the compaction pressure implied by the current code variables if pr is
    interpreted as total pressure P and pf as fluid pressure Pf.  solidP and
    solidB are the solid volume fraction, max(1-phi, phimin), on P and basic
    nodes.
    """
    st.Pc[...] = st.pr - st.pf
    copy_pnode_edges(st.Pc)

    st.solidP[...] = np.maximum(1.0 - np.clip(st.PHI, 0.0, 1.0), 0.0)
    copy_pnode_edges(st.solidP)

    phiB = 0.25 * (
        st.PHI[0:cfg.ny, 0:cfg.nx]
        + st.PHI[1:cfg.ny + 1, 0:cfg.nx]
        + st.PHI[0:cfg.ny, 1:cfg.nx + 1]
        + st.PHI[1:cfg.ny + 1, 1:cfg.nx + 1]
    )
    st.solidB[...] = np.maximum(1.0 - np.clip(phiB, 0.0, 1.0), 0.0)


def apply_low_phi_darcy_cutoff(st: State, cfg: Config) -> None:
    """Apply the Keller Appendix-A4 permeability endpoint rules.

    At phi=0 or below the under-connected threshold, keep the non-zero
    stabilization permeability kphi_min. At true full melt, the solid
    framework disappears and the phase-separation Darcy flux is zero.
    """
    low_x = st.PHIX < cfg.phi_dry_crit
    low_y = st.PHIY < cfg.phi_dry_crit
    st.KXOE[low_x] = cfg.kphi_min / cfg.etafluid
    st.KYOE[low_y] = cfg.kphi_min / cfg.etafluid

    full_x = st.PHIX >= cfg.phi_full_crit
    full_y = st.PHIY >= cfg.phi_full_crit
    st.KXOE[full_x] = 0.0
    st.KYOE[full_y] = 0.0
    st.qxD[full_x] = 0.0
    st.qyD[full_y] = 0.0


def apply_low_phi_pressure_projection(st: State, cfg: Config, *, include_history: bool = False) -> None:
    """Project both single-phase endpoints to the one-pressure limit Pc=0.

    Dry/under-connected material and true full melt cannot sustain an
    independent compaction pressure, so Pf=P. The low-phi permeability is not
    zeroed here; it remains at kphi_min for pressure stabilization.
    """
    dry_or_underconnected = st.PHI < cfg.phi_dry_crit
    full = st.PHI >= cfg.phi_full_crit
    endpoint_p = dry_or_underconnected | full
    st.pf[endpoint_p] = st.pr[endpoint_p]
    if include_history:
        st.pf0[endpoint_p] = st.pr0[endpoint_p]
    copy_pnode_edges(st.pf)
    if include_history:
        copy_pnode_edges(st.pf0)


def apply_velocity_bc(st: State, cfg: Config) -> None:
    # Keller Fig. 3/Section 3.1 boundary conditions:
    #   * side boundaries: imposed extensional normal velocity; zero tangential stress
    #   * bottom boundary: shear-stress free / free-slip approximation
    #   * top boundary: stress-free approximation
    #
    # In this staggered-grid code the stress-free/free-slip conditions are imposed
    # as zero-gradient ghost/edge values for the tangential velocity component.
    # The top boundary no longer clamps vy=0, because Keller states that the top
    # boundary is stress-free, not no-penetration.
    vxleft = -cfg.strainrate * cfg.xsize / 2.0
    vxright = cfg.strainrate * cfg.xsize / 2.0
    nx = cfg.nx
    ny = cfg.ny

    # Left/right: prescribed normal velocity and zero tangential stress.
    st.vx[:, 0] = vxleft
    st.vx[:, nx - 1] = vxright
    st.vx[:, nx] = 0.0
    st.vy[1:ny - 1, 0] = st.vy[1:ny - 1, 1]
    st.vy[1:ny - 1, nx] = st.vy[1:ny - 1, nx - 1]

    # Bottom: no penetration plus zero tangential stress.
    st.vy[ny - 1, :] = 0.0
    st.vy[ny, :] = 0.0
    st.vx[ny, 1:nx - 1] = st.vx[ny - 1, 1:nx - 1]

    # Outer top: traction (stress-free, anchored to cfg.surface_pressure)
    # boundary -- aligned with B's half-cell top momentum balance instead of
    # the old impermeable Dirichlet condition vy_top = 0.
    # Interior top-row columns (1..nx-1) remain PT degrees of freedom and must
    # not be overwritten here.  Only the two outer corners are mirrored.
    st.vy[0, 0] = st.vy[0, 1]
    st.vy[0, nx] = st.vy[0, nx - 1]

    st.vx[0, 1:nx - 1] = st.vx[1, 1:nx - 1]


def apply_hydraulic_bc(st: State, cfg: Config) -> None:
    nx = cfg.nx
    ny = cfg.ny

    # Pure Pf boundary-condition version.
    # First keep the usual P-node ghost rows/columns consistent for all fields.
    # Then overwrite only the lower source segment of the fluid-pressure ghost
    # row with the prescribed reservoir pressure.  The Darcy update then computes
    # qyD[ny-1, source] from this Pf gradient; enforce_darcy_flux_bc() no longer
    # computes or clips a separate source qyD.
    for arr in (st.RHO, st.pr, st.pf, st.PHI, st.SXX, st.SYY, st.GGGP, st.ETAP, st.XI, st.XI0, st.BETTAPHI, st.APHI, st.DSYT):
        copy_pnode_edges(arr)
    copy_pnode_edges(st.YNYT)

    if cfg.source_halfwidth > 0.0:
        c = coords(cfg)
        source_p = np.abs(c["xp"] - cfg.source_x) <= cfg.source_halfwidth
        # Pf is cell-centred.  Enforce the prescribed value at the physical
        # bottom face halfway between the last interior and ghost P nodes.
        st.pf[ny, source_p] = (
            2.0 * cfg.source_pressure - st.pf[ny - 1, source_p]
        )


def enforce_darcy_flux_bc(st: State, cfg: Config) -> None:
    """Darcy-flux boundary cleanup for the pure Pf source-boundary version.

    Closed boundaries are kept no-flux.  On the lower source segment, qyD is not
    imposed here.  Instead, apply_hydraulic_bc() prescribes Pf on the bottom
    ghost row, and the staggered Darcy update computes qyD[ny-1, source] from
    the resulting pressure gradient.
    """
    nx = cfg.nx
    ny = cfg.ny
    c = coords(cfg)

    # Left and right: no horizontal Darcy flux.
    st.qxD[:, 0] = 0.0
    st.qxD[:, nx - 1] = 0.0
    st.qxD[:, nx] = 0.0

    # Top: no vertical Darcy flux.
    st.qyD[0, :] = 0.0

    # Bottom non-source: no vertical Darcy flux.  The source segment is left
    # untouched because it is computed by the Darcy update from the prescribed
    # Pf ghost-row boundary value.
    source_q = np.abs(c["xvy"] - cfg.source_x) <= cfg.source_halfwidth
    st.qyD[ny - 1, ~source_q] = 0.0

    # qyD[ny, :] is a ghost row and is not used as the physical source flux.
    st.qyD[ny, :] = 0.0


def initialize_lithostatic_pressure(st: State, cfg: Config) -> None:
    """Initialize Pt=Pf from the air-plus-rock integrated density column.

    The top ``sticky_air_thickness`` is assigned rho_air.  Below the initial
    air-rock interface, the rock/melt mixture density is integrated using the
    same analytic Gaussian porosity field as the lower source pulse.
    """
    c = coords(cfg)
    depth = np.clip(c["yp"], 0.0, cfg.ysize)[:, None]
    xp = c["xp"][None, :]

    air_depth = np.minimum(depth, cfg.sticky_air_thickness)
    rock_depth = np.maximum(depth - cfg.sticky_air_thickness, 0.0)

    horizontal = np.exp(
        -((xp - cfg.source_x) ** 2)
        / (2.0 * cfg.gaussian_sigma_x**2)
    )

    # Integrate the vertical Gaussian only through the rock, from the initial
    # air-rock interface to the requested depth.  For points in sticky air the
    # integral is exactly zero.
    rock_upper = np.maximum(depth[:, 0], cfg.sticky_air_thickness)
    scaled_upper = (
        rock_upper - cfg.ysize
    ) / (math.sqrt(2.0) * cfg.gaussian_sigma_y)
    erf_upper = np.fromiter(
        (math.erf(float(value)) for value in scaled_upper),
        dtype=np.float64,
        count=scaled_upper.size,
    )[:, None]
    erf_rock_top = math.erf(
        (cfg.sticky_air_thickness - cfg.ysize)
        / (math.sqrt(2.0) * cfg.gaussian_sigma_y)
    )
    gaussian_column_integral = (
        cfg.gaussian_sigma_y
        * math.sqrt(math.pi / 2.0)
        * (erf_upper - erf_rock_top)
    )
    gaussian_column_integral[depth <= cfg.sticky_air_thickness] = 0.0

    phi_column_integral = (
        cfg.phi_background * rock_depth
        + cfg.phi_amplitude * horizontal * gaussian_column_integral
    )

    pressure = cfg.surface_pressure + cfg.gravity * (
        cfg.rho_air * air_depth
        + cfg.rho_solid * rock_depth
        - (cfg.rho_solid - cfg.rho_fluid) * phi_column_integral
    )

    st.pr[:, :] = pressure
    st.pf[:, :] = pressure
    st.pr0[:, :] = pressure
    st.pf0[:, :] = pressure


def initial_state(cfg: Config) -> State:
    bshape = (cfg.ny, cfg.nx)
    pshape = (cfg.ny1, cfg.nx1)
    xm, ym, tm, phim, etavpm, xivpm, sxxm, syym, sxym = initial_markers(cfg)
    st = State(
        ETA=zeros(bshape), ETA0=zeros(bshape), GGG=np.full(bshape, cfg.G0),
        EXY=zeros(bshape), SXY=zeros(bshape), SXY0=zeros(bshape),
        COH=np.full(bshape, cfg.coh0), TEN=np.full(bshape, cfg.tens0), FRI=zeros(bshape),
        YNY=np.zeros(bshape, dtype=bool), SIIB=zeros(bshape), SYIELD=zeros(bshape), DSY=zeros(bshape), wyx=zeros(bshape),
        vx=zeros(pshape), qxD=zeros(pshape), PHIX=np.full(pshape, cfg.phi_background), KXOE=np.full(pshape, cfg.kphi_background / cfg.etafluid),
        vy=zeros(pshape), qyD=zeros(pshape), PHIY=np.full(pshape, cfg.phi_background), KYOE=np.full(pshape, cfg.kphi_background / cfg.etafluid),
        RHO=np.full(pshape, cfg.rho_solid),
        pr=zeros(pshape), pf=zeros(pshape), pr0=zeros(pshape), pf0=zeros(pshape),
        PHI=np.full(pshape, cfg.phi_background), PHI0=np.full(pshape, cfg.phi_background),
        BETTAPHI=np.full(pshape, cfg.beta_phi_background), ETAP=zeros(pshape), XI=zeros(pshape),
        XI0=zeros(pshape),
        YNYT=np.zeros(pshape, dtype=bool), DSYT=zeros(pshape),
        GGGP=np.full(pshape, cfg.G0),
        EXX=zeros(pshape), EYY=zeros(pshape),
        SXX=zeros(pshape), SXX0=zeros(pshape), SYY=zeros(pshape), SYY0=zeros(pshape),
        DIVV=zeros(pshape), APHI=zeros(pshape),
        dphidt=zeros(pshape), pf_prev_iter=zeros(pshape),
        Pc=zeros(pshape), solidP=np.ones(pshape), solidB=np.ones(bshape),
        vxp=zeros(pshape), vyp=zeros(pshape),
        vydif=zeros(pshape), vy_prev_iter=zeros(pshape),
        xm=xm, ym=ym, tm=tm, phim=phim, etavpm=etavpm, xivpm=xivpm,
        sxxm=sxxm, syym=syym, sxym=sxym,
    )
    interpolate_markers_to_nodes(st, cfg, with_stress=True)
    initialize_lithostatic_pressure(st, cfg)

    c = coords(cfg)
    # Horizontal extensional starting field consistent with the side velocities.
    vx_profile = cfg.strainrate * (c["xvx"] - cfg.xsize / 2.0)
    vy_profile = np.zeros_like(c["yvy"])
    st.vx[:, :] = vx_profile[None, :]
    st.vy[:, :] = vy_profile[:, None]
    apply_velocity_bc(st, cfg)
    apply_hydraulic_bc(st, cfg)
    enforce_darcy_flux_bc(st, cfg)
    compute_etap(st, cfg)
    update_keller_step1_diagnostics(st, cfg)
    return st


def interpolate_markers_to_nodes(st: State, cfg: Config, *, with_stress: bool) -> None:
    """MOD4: align marker->node nonlinear properties with B, keep D air mixing.

    Rock-only nodes use B's grid-first order: marker phi -> nodal phi -> eta/xi/k.
    ETA/XI carried histories are capped by the current nodal unyielded laws.
    At nodes touched by sticky-air markers, ETA0/XI0 deliberately retain D's
    original arithmetic marker-property mixing so that air mixing itself is NOT
    changed in this experiment.  Permeability is porosity-only, so it follows
    B's grid-first PHIX/PHIY evaluation everywhere.
    """
    recharge_source_markers(st, cfg)

    airm = (st.tm == 3).astype(float)
    eta0m = material_matrix_viscosity(st.tm, st.phim, cfg)
    xiphi0m = material_xi0_viscosity(st.tm, st.phim, cfg)
    rhom = marker_density(st.tm, st.phim, cfg)
    fricm = np.where(st.tm == 1, cfg.fric_block, 0.0)
    cohm = np.full_like(st.xm, cfg.coh0)
    tenm = np.full_like(st.xm, cfg.tens0)
    cohm[st.tm == 3] = 0.0
    tenm[st.tm == 3] = 0.0
    invGm = np.full_like(st.xm, 1.0 / cfg.G0)

    # Basic nodes: B grid-first rock law, but D arithmetic mixing wherever air contributes.
    bshape = (cfg.ny, cfg.nx)
    acc_phi, w = scatter_to_grid(st.xm, st.ym, st.phim, bshape, cfg, x0=0.0, y0=0.0)
    phiB = np.clip(average_scatter(acc_phi, w, cfg.phi_background), 0.0, 1.0)
    acc_air, _ = scatter_to_grid(st.xm, st.ym, airm, bshape, cfg, x0=0.0, y0=0.0)
    airB = np.clip(average_scatter(acc_air, w, 0.0), 0.0, 1.0)

    eta_grid_B = material_matrix_viscosity(np.ones(bshape, dtype=np.int64), phiB, cfg)
    acc_eta0_D, _ = scatter_to_grid(st.xm, st.ym, eta0m, bshape, cfg, x0=0.0, y0=0.0)
    eta0_D_airmix = average_scatter(acc_eta0_D, w, cfg.eta_background)
    st.ETA0[...] = np.where(airB > 0.0, eta0_D_airmix, eta_grid_B)

    acc_eta_hist, _ = scatter_to_grid(st.xm, st.ym, st.etavpm, bshape, cfg, x0=0.0, y0=0.0)
    st.ETA[...] = np.minimum(average_scatter(acc_eta_hist, w, st.ETA0), st.ETA0)
    st.YNY[...] = st.ETA < st.ETA0

    acc, _ = scatter_to_grid(st.xm, st.ym, invGm, bshape, cfg, x0=0.0, y0=0.0)
    invG = average_scatter(acc, w, 1.0 / cfg.G0)
    st.GGG[...] = 1.0 / np.maximum(invG, 1.0e-300)
    acc, _ = scatter_to_grid(st.xm, st.ym, cohm, bshape, cfg, x0=0.0, y0=0.0)
    st.COH[...] = average_scatter(acc, w, cfg.coh0)
    acc, _ = scatter_to_grid(st.xm, st.ym, tenm, bshape, cfg, x0=0.0, y0=0.0)
    st.TEN[...] = average_scatter(acc, w, cfg.tens0)
    acc, _ = scatter_to_grid(st.xm, st.ym, fricm, bshape, cfg, x0=0.0, y0=0.0)
    st.FRI[...] = average_scatter(acc, w, 0.0)
    if with_stress:
        acc, _ = scatter_to_grid(st.xm, st.ym, st.sxym, bshape, cfg, x0=0.0, y0=0.0)
        st.SXY0[...] = average_scatter(acc, w, 0.0)
        st.SXY[...] = st.SXY0

    # Vx/Vy nodes: B grid-first permeability/mobility from nodal staggered phi.
    pshape = (cfg.ny1, cfg.nx1)
    acc, wvx = scatter_to_grid(st.xm, st.ym, st.phim, pshape, cfg, x0=0.0, y0=-cfg.dy / 2.0, max_i0=cfg.ny - 1, max_j0=cfg.nx - 2)
    st.PHIX[...] = np.clip(average_scatter(acc, wvx, cfg.phi_background), 0.0, 1.0)
    st.KXOE[...] = marker_permeability_over_eta(np.ones_like(st.PHIX), st.PHIX, cfg)

    acc, wvy = scatter_to_grid(st.xm, st.ym, st.phim, pshape, cfg, x0=-cfg.dx / 2.0, y0=0.0, max_i0=cfg.ny - 2, max_j0=cfg.nx - 1)
    st.PHIY[...] = np.clip(average_scatter(acc, wvy, cfg.phi_background), 0.0, 1.0)
    st.KYOE[...] = marker_permeability_over_eta(np.ones_like(st.PHIY), st.PHIY, cfg)

    # P nodes: B grid-first rock xi law, but D arithmetic mixing wherever air contributes.
    acc_phi_p, wp = scatter_to_grid(st.xm, st.ym, st.phim, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    acc_rho, _ = scatter_to_grid(st.xm, st.ym, rhom, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    st.RHO[...] = average_scatter(acc_rho, wp, cfg.rho_solid)
    copy_pnode_edges(st.RHO)
    st.PHI[...] = np.clip(average_scatter(acc_phi_p, wp, cfg.phi_background), 0.0, 1.0)

    acc_air_p, _ = scatter_to_grid(st.xm, st.ym, airm, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    airP = np.clip(average_scatter(acc_air_p, wp, 0.0), 0.0, 1.0)
    xi_grid_B = material_xi0_viscosity(np.ones(pshape, dtype=np.int64), st.PHI, cfg)
    acc_xi0_D, _ = scatter_to_grid(st.xm, st.ym, xiphi0m, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    xi0_D_airmix = average_scatter(acc_xi0_D, wp, cfg.xi_background)
    st.XI0[...] = np.where(airP > 0.0, xi0_D_airmix, xi_grid_B)
    copy_pnode_edges(st.XI0)

    acc_xi_hist, _ = scatter_to_grid(st.xm, st.ym, st.xivpm, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    st.XI[...] = np.minimum(average_scatter(acc_xi_hist, wp, st.XI0), st.XI0)
    copy_pnode_edges(st.XI)
    st.YNYT[...] = st.XI < st.XI0
    copy_pnode_edges(st.YNYT)

    acc, _ = scatter_to_grid(st.xm, st.ym, invGm, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    invGP = average_scatter(acc, wp, 1.0 / cfg.G0)
    st.GGGP[...] = 1.0 / np.maximum(invGP, 1.0e-300)
    update_bettaphi_keller(st, cfg)
    if with_stress:
        acc, _ = scatter_to_grid(st.xm, st.ym, st.sxxm, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
        st.SXX0[...] = average_scatter(acc, wp, 0.0)
        st.SXX[...] = st.SXX0
        acc, _ = scatter_to_grid(st.xm, st.ym, st.syym, pshape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
        st.SYY0[...] = average_scatter(acc, wp, 0.0)
        st.SYY[...] = st.SYY0

    apply_source_reservoir_nodes(st, cfg)
    apply_low_phi_darcy_cutoff(st, cfg)
    compute_etap(st, cfg)
    apply_hydraulic_bc(st, cfg)
    update_keller_step1_diagnostics(st, cfg)


def compute_etap(st: State, cfg: Config) -> None:
    """Update ETAP, the P-node effective shear viscosity.

    ETAP is the current/effective shear viscosity averaged from ETA with the
    same four-node harmonic average as the reference MATLAB code.  It is used
    by the deviatoric SXX Maxwell update because ETA may already include
    shear-plastic weakening.

    XI0 and XI are already P-node compaction viscosities obtained from
    marker-to-node interpolation in interpolate_markers_to_nodes().  This
    function deliberately does not modify XI, because XI plays the same
    state-variable role for compaction plasticity that ETA plays for shear
    plasticity.
    """
    ny, nx = cfg.ny, cfg.nx

    denom_eff = (
        1.0 / np.maximum(st.ETA[:-1, :-1], 1.0e-300)
        + 1.0 / np.maximum(st.ETA[1:, :-1], 1.0e-300)
        + 1.0 / np.maximum(st.ETA[:-1, 1:], 1.0e-300)
        + 1.0 / np.maximum(st.ETA[1:, 1:], 1.0e-300)
    )
    st.ETAP[1:ny, 1:nx] = 4.0 / np.maximum(denom_eff, 1.0e-300)
    copy_pnode_edges(st.ETAP)


def snapshot_state(st: State) -> dict[str, np.ndarray]:
    return {name: getattr(st, name).copy() for name in State.__dataclass_fields__ if isinstance(getattr(st, name), np.ndarray)}


def restore_state(st: State, snap: dict[str, np.ndarray]) -> None:
    for name, arr in snap.items():
        getattr(st, name)[...] = arr


def set_eta_yny(st: State, ETA: np.ndarray, YNY: np.ndarray) -> None:
    st.ETA[...] = ETA
    st.YNY[...] = YNY



def set_tensile_yny(st: State, XI: np.ndarray, YNYT: np.ndarray) -> None:
    st.XI[...] = XI
    st.YNYT[...] = YNYT
    copy_pnode_edges(st.XI)
    copy_pnode_edges(st.YNYT)


def solve_hm_fixed_eta(
    st: State,
    cfg: Config,
    it: int,
    dt: float,
    Vpdt: float,
    Kfdt: float,
    rhof_dt: float,
    *,
    iplast: int = 0,
    runtime_logger: RuntimeLogger | None = None,
) -> tuple[int, float, float, bool]:
    ny, nx = cfg.ny, cfg.nx
    dx, dy = cfg.dx, cfg.dy
    compute_etap(st, cfg)
    resid = 2.0 * cfg.epsi
    err_v = 0.0
    err_vy = 0.0
    err_pf_diff = 0.0
    err_pf = 0.0
    st.vy_prev_iter[...] = st.vy
    Krf = Kfdt / (cfg.Ks * dt)

    for itpt in range(cfg.niter):
        eta_ve = np.nanmax(1.0 / (1.0 / np.maximum(st.ETA, 1.0e-300) + 1.0 / np.maximum(st.GGG * dt, 1.0e-300)))
        dt_rho = Vpdt * cfg.xsize / (cfg.Re * eta_ve)
        Gdt = Vpdt * Vpdt / (dt_rho * (1.0 + 4.0 / 3.0))
        Kdt = Gdt
        Kr = Kdt / (cfg.Ks * dt)

        # P-node strain rate and pressure updates, interior i=2:Ny, j=2:Nx in MATLAB.
        ip = slice(1, ny)
        jp = slice(1, nx)
        dvx_dx = (st.vx[ip, jp] - st.vx[ip, 0:nx - 1]) / dx
        dvy_dy = (st.vy[ip, jp] - st.vy[0:ny - 1, jp]) / dy
        st.DIVV[ip, jp] = dvx_dx + dvy_dy
        # 3-D deviatoric projection in a 2-D plane-strain calculation,
        # matching the nondimensional porosity-wave code:
        # D'xx = Dxx - div(v)/3, D'yy = Dyy - div(v)/3,
        # with the implicit out-of-plane D'zz = -div(v)/3.
        st.EXX[ip, jp] = dvx_dx - st.DIVV[ip, jp] / 3.0
        st.EYY[ip, jp] = dvy_dy - st.DIVV[ip, jp] / 3.0
        apply_low_phi_darcy_cutoff(st, cfg)
        divQ = (st.qxD[ip, jp] - st.qxD[ip, 0:nx - 1]) / dx + (st.qyD[ip, jp] - st.qyD[0:ny - 1, jp]) / dy

        st.pf_prev_iter[...] = st.pf
        phi_state, _, solid_frac = material_phase_fractions(st.PHI[ip, jp], cfg)
        dryP = phi_state < cfg.phi_dry_crit
        fullP = phi_state >= cfg.phi_full_crit
        endpointP = dryP | fullP
        etaphi = np.maximum(st.XI[ip, jp], 1.0e-300)
        dphidt_local = (st.pf[ip, jp] - st.pr[ip, jp]) / etaphi + st.BETTAPHI[ip, jp] / dt * (
            (st.pf[ip, jp] - st.pf0[ip, jp]) - (st.pr[ip, jp] - st.pr0[ip, jp])
        )
        dphidt_local[endpointP] = 0.0
        st.dphidt[ip, jp] = dphidt_local
        pr_new = (st.pr[ip, jp] + st.pr0[ip, jp] * Kr - Kdt * (st.DIVV[ip, jp] - st.dphidt[ip, jp] / solid_frac)) / (1.0 + Kr)
        pf_new = (st.pf[ip, jp] + st.pf0[ip, jp] * Krf - (divQ + (st.pf[ip, jp] - pr_new) / etaphi / solid_frac) /
                  (1.0 / Kfdt + 1.0 / etaphi / solid_frac)) / (1.0 + Krf)
        pf_new[endpointP] = pr_new[endpointP]
        st.pr[ip, jp] = pr_new
        st.pf[ip, jp] = pf_new

        # Visco-elastic normal deviatoric stresses at P nodes.  SXX and SYY
        # are independent history variables; do not impose SYY = -SXX when
        # div(v_s) is non-zero.
        GrP = Gdt / np.maximum(st.GGGP[ip, jp] * dt, 1.0e-300)
        denomP = 1.0 / Gdt + 1.0 / np.maximum(st.ETAP[ip, jp], 1.0e-300) + GrP / Gdt
        st.SXX[ip, jp] += (
            -(st.SXX[ip, jp] - st.SXX0[ip, jp]) * GrP / Gdt
            - st.SXX[ip, jp] / np.maximum(st.ETAP[ip, jp], 1.0e-300)
            + 2.0 * st.EXX[ip, jp]
        ) / denomP
        st.SYY[ip, jp] += (
            -(st.SYY[ip, jp] - st.SYY0[ip, jp]) * GrP / Gdt
            - st.SYY[ip, jp] / np.maximum(st.ETAP[ip, jp], 1.0e-300)
            + 2.0 * st.EYY[ip, jp]
        ) / denomP

        # Basic-node shear strain rate and SXY update.
        st.EXY[:, :] = 0.5 * ((st.vx[1:ny + 1, 0:nx] - st.vx[0:ny, 0:nx]) / dy + (st.vy[0:ny, 1:nx + 1] - st.vy[0:ny, 0:nx]) / dx)
        GrB = Gdt / np.maximum(st.GGG * dt, 1.0e-300)
        denomB = 1.0 / Gdt + 1.0 / np.maximum(st.ETA, 1.0e-300) + GrB / Gdt
        st.SXY += (
            -(st.SXY - st.SXY0) * GrB / Gdt
            - st.SXY / np.maximum(st.ETA, 1.0e-300)
            + 2.0 * st.EXY
        ) / denomB

        apply_hydraulic_bc(st, cfg)

        # Darcy flux updates on their staggered nodes.
        qx_rows = slice(1, ny)
        qx_cols = slice(1, nx - 1)
        st.qxD[qx_rows, qx_cols] += (
            -st.qxD[qx_rows, qx_cols]
            - st.KXOE[qx_rows, qx_cols] * (st.pf[qx_rows, 2:nx] - st.pf[qx_rows, 1:nx - 1]) / dx
        ) / (1.0 + st.KXOE[qx_rows, qx_cols] * rhof_dt)
        qy_rows = slice(1, ny)
        qy_cols = slice(1, nx)
        st.qyD[qy_rows, qy_cols] += (
            -st.qyD[qy_rows, qy_cols]
            - st.KYOE[qy_rows, qy_cols] * (
                (st.pf[2:ny + 1, qy_cols] - st.pf[1:ny, qy_cols]) / dy
                - cfg.rho_fluid * cfg.gravity
            )
        ) / (1.0 + st.KYOE[qy_rows, qy_cols] * rhof_dt)
        apply_low_phi_darcy_cutoff(st, cfg)
        enforce_darcy_flux_bc(st, cfg)

        # Momentum updates.  These are P-T updates of the same staggered force
        # balances used by the MATLAB matrix assembly.
        #
        # Keller-style total deviatoric stress in the bulk momentum equation:
        # keep SXX/SYY/SXY themselves as solid-matrix stresses for the Maxwell
        # history, marker storage, and rotation.  Away from the full-melt
        # endpoint use tau_total = (1 - phi) * tau_solid.  In the full-melt
        # limit, replace it by Keller's high-melt cut-off approximation
        # tau_total ~= 2*eta_cutoff*strain_rate, so the equation tends to a
        # regularized single-fluid Stokes limit instead of losing all shear
        # resistance as (1 - phi) -> 0.
        solidP_mom = 1.0 - st.PHI
        SXX_total = solidP_mom * st.SXX
        SYY_total = solidP_mom * st.SYY
        fullP_mom = st.PHI >= cfg.phi_full_crit
        SXX_total[fullP_mom] = 2.0 * cfg.eta_melt_cutoff * st.EXX[fullP_mom]
        SYY_total[fullP_mom] = 2.0 * cfg.eta_melt_cutoff * st.EYY[fullP_mom]

        phiB_mom = p_to_basic_average(st.PHI, cfg)
        solidB_mom = 1.0 - phiB_mom
        SXY_total = solidB_mom * st.SXY
        fullB_mom = phiB_mom >= cfg.phi_full_crit
        SXY_total[fullB_mom] = 2.0 * cfg.eta_melt_cutoff * st.EXY[fullB_mom]

        vxs = (slice(1, ny), slice(1, nx - 1))
        force_x = (
            (SXX_total[1:ny, 2:nx] - SXX_total[1:ny, 1:nx - 1] - st.pr[1:ny, 2:nx] + st.pr[1:ny, 1:nx - 1]) / dx
            + (SXY_total[1:ny, 1:nx - 1] - SXY_total[0:ny - 1, 1:nx - 1]) / dy
        )
        st.vx[vxs] += dt_rho * force_x

        vys = (slice(1, ny - 1), slice(1, nx))
        rho_y = 0.5 * (st.RHO[2:ny, 1:nx] + st.RHO[1:ny - 1, 1:nx])
        force_y = (
            (SYY_total[2:ny, 1:nx] - SYY_total[1:ny - 1, 1:nx] - st.pr[2:ny, 1:nx] + st.pr[1:ny - 1, 1:nx]) / dy
            + (SXY_total[1:ny - 1, 1:nx] - SXY_total[1:ny - 1, 0:nx - 1]) / dx
            + rho_y * cfg.gravity
        )
        st.vy[vys] += dt_rho * force_y
        st.vydif[vys] = st.vy[vys] - st.vy_prev_iter[vys]
        st.vy_prev_iter[vys] = st.vy[vys]

        # Outer top row (i=0): B-style half-cell traction balance,
        #   2/dy*(SYY_total(1,j) - pr(1,j) + P_surface)
        #   + d/dx SXY_total(0,j) + rho_air*g = 0.
        # The top normal velocity is relaxed as an unknown instead of clamped.
        force_y_top = (
            2.0 / dy * (SYY_total[1, 1:nx] - st.pr[1, 1:nx] + cfg.surface_pressure)
            + (SXY_total[0, 1:nx] - SXY_total[0, 0:nx - 1]) / dx
            + cfg.rho_air * cfg.gravity
        )
        st.vy[0, 1:nx] += dt_rho * force_y_top

        apply_velocity_bc(st, cfg)
        apply_hydraulic_bc(st, cfg)

        itpt_one_based = itpt + 1
        check_now = (4 < itpt < 50) or (itpt > 49 and itpt % cfg.nout == 1)
        csv_log_now = runtime_logger is not None and runtime_logger.should_log_pt(itpt_one_based)
        last_iter_now = itpt == cfg.niter - 1
        final_console_enabled = runtime_logger is not None and runtime_logger.should_print_final_pt()
        if check_now or csv_log_now or last_iter_now:
            err_v = float(np.nanmax(np.abs(st.vydif[vys])))
            err_vy = float(np.nanmax(np.abs(st.vy)))
            err_pf_diff = float(np.nanmax(np.abs(st.pf_prev_iter - st.pf)))
            err_pf = float(np.nanmax(np.abs(st.pf)))
            resid = err_v / (1.0 + err_vy) + err_pf_diff / max(err_pf, 1.0e-300)
            converged_now = resid <= cfg.epsi
            print_console_now = final_console_enabled and (last_iter_now or (check_now and converged_now))
            if csv_log_now or print_console_now:
                runtime_logger.log_pt(
                    st,
                    cfg,
                    step=it,
                    iplast=iplast,
                    itpt=itpt_one_based,
                    dt=dt,
                    dt_rho=dt_rho,
                    Gdt=Gdt,
                    Kdt=Kdt,
                    resid=resid,
                    err_v=err_v,
                    err_vy=err_vy,
                    err_pf_diff=err_pf_diff,
                    err_pf=err_pf,
                    converged=converged_now,
                    write_csv=csv_log_now,
                    print_console=print_console_now,
                )
            if check_now and converged_now:
                update_keller_step1_diagnostics(st, cfg)
                return itpt + 1, resid, err_vy, True

    update_keller_step1_diagnostics(st, cfg)
    return cfg.niter, resid, err_vy, resid <= cfg.epsi


def p_to_basic_average(pfield: np.ndarray, cfg: Config) -> np.ndarray:
    ny, nx = cfg.ny, cfg.nx
    return 0.25 * (pfield[0:ny, 0:nx] + pfield[1:ny + 1, 0:nx] + pfield[0:ny, 1:nx + 1] + pfield[1:ny + 1, 1:nx + 1])


def basic_to_p_average(bfield: np.ndarray, cfg: Config) -> np.ndarray:
    """Average a basic-node field to P nodes."""
    ny, nx = cfg.ny, cfg.nx
    out = np.zeros((cfg.ny1, cfg.nx1), dtype=np.float64)
    out[1:ny, 1:nx] = 0.25 * (
        bfield[0:ny - 1, 0:nx - 1]
        + bfield[1:ny, 0:nx - 1]
        + bfield[0:ny - 1, 1:nx]
        + bfield[1:ny, 1:nx]
    )
    copy_pnode_edges(out)
    return out


def compute_plastic_active_set(st: State, cfg: Config, dt: float) -> PlasticResult:
    # SXX/SYY/SXY are stored as solid-matrix deviatoric stresses so that the
    # Maxwell history, marker storage, and rotation remain solid-stress based.
    # Keller's yield functions, however, use the mixture/bulk deviatoric stress
    # invariant tau_II, with tau = (1 - phi) * tau_s.  Therefore SIIB is the
    # total-stress invariant at basic nodes.
    SXXB_solid = p_to_basic_average(st.SXX, cfg)
    SYYB_solid = p_to_basic_average(st.SYY, cfg)
    tauII_solid = np.sqrt(0.5 * (SXXB_solid * SXXB_solid + SYYB_solid * SYYB_solid) + st.SXY * st.SXY)
    phiB = p_to_basic_average(st.PHI, cfg)
    solidB = 1.0 - phiB
    st.SIIB[...] = solidB * tauII_solid

    prB = p_to_basic_average(st.pr, cfg)
    pfB = p_to_basic_average(st.pf, cfg)

    # Keller-style effective-pressure switch.  In dry/under-connected regions
    # (phi < phi_crit), use Pe = P.  Once enough melt is present, use
    # Terzaghi effective pressure Pe = P - Pf.  Here pr is interpreted as
    # the total/bulk pressure P.
    xphiB = (phiB >= cfg.phi_dry_crit).astype(np.float64)
    peff = prB - xphiB * pfB

    # Split the Keller yield envelope by branch.  Shear viscosity ETA is
    # weakened only on the Mohr-Coulomb branch, Pe > Pe*.  The tensile/
    # Griffith branch is handled separately by compute_tensile_plastic_active_set()
    # through XI, so do not also reduce ETA there.
    sin_phi = np.clip(st.FRI, 0.0, 1.0 - 1.0e-12)
    cos_phi = np.sqrt(np.maximum(1.0 - sin_phi * sin_phi, 0.0))
    shear_branch = st.COH * cos_phi + peff * sin_phi
    pe_star = (st.COH * cos_phi - st.TEN) / np.maximum(1.0 - sin_phi, 1.0e-12)
    fullB = phiB >= cfg.phi_full_crit
    shear_branch_active = (peff > pe_star) & (~fullB)

    st.SYIELD.fill(np.inf)
    st.SYIELD[shear_branch_active] = np.maximum(shear_branch[shear_branch_active], 0.0)

    eta_safe = np.maximum(st.ETA, 1.0e-300)
    siiel = st.SIIB * (st.GGG * dt + eta_safe) / eta_safe
    ETA5 = st.ETA0.copy()
    YNY5 = np.zeros_like(st.YNY)
    st.DSY.fill(0.0)

    old = st.YNY.copy()
    old_active = old & shear_branch_active
    ynpl = int(np.count_nonzero(old_active))
    ddd = float(np.sum((st.SIIB[old_active] - st.SYIELD[old_active]) ** 2)) if ynpl > 0 else 0.0
    st.DSY[old_active] = st.SIIB[old_active] - st.SYIELD[old_active]

    may = shear_branch_active & (st.SYIELD < siiel)
    etapl = np.full_like(st.ETA, np.inf)
    etapl[may] = dt * st.GGG[may] * st.SYIELD[may] / np.maximum(siiel[may] - st.SYIELD[may], 1.0e-300)
    new = may & (etapl < st.ETA0)
    if np.any(new):
        ETA5[new] = (etapl[new] ** (1.0 - cfg.etawt)) * (st.ETA[new] ** cfg.etawt)
        ETA5[new] = np.clip(ETA5[new], cfg.eta_min, cfg.eta_max)
        YNY5[new] = True
        add = new & (~old)
        if np.any(add):
            st.DSY[add] = st.SIIB[add] - st.SYIELD[add]
            ddd += float(np.sum(st.DSY[add] ** 2))
            ynpl += int(np.count_nonzero(add))

    yerr = math.sqrt(ddd / ynpl) if ynpl > 0 else 0.0
    return PlasticResult(
        ETA5=ETA5,
        YNY5=YNY5,
        yerr=yerr,
        ynpl=ynpl,
        yny_old_count=int(np.count_nonzero(old)),
        yny5_count=int(np.count_nonzero(YNY5)),
        yny_changed_count=int(np.count_nonzero(old != YNY5)),
    )


def compute_tensile_plastic_active_set(st: State, cfg: Config, dt: float) -> TensilePlasticResult:
    """Active-set update for Keller-style tensile/compaction yielding.

    SXX/SYY/SXY are stored as solid-matrix stresses.  For the tensile yield
    pressure, use the Keller mixture-stress invariant
    tau_II = (1 - phi) * tau_s,II at P nodes and
    P_y = tau_II - sigma_T.  The compaction pressure is Pc = pr - pf.

    The return mapping mirrors the shear update:
        Pc_trial = Pc * (K_phi*dt + xi) / xi,
        if Pc_trial < P_y, xi_pl = K_phi*dt*P_y/(Pc_trial - P_y).
    This simplified trial-return form is valid for the usual tensile-
    overpressure branch with P_y < 0.  Points with Pc_trial < P_y but
    P_y >= 0 are recorded as invalid for this return map and are not weakened.
    """
    ny, nx = cfg.ny, cfg.nx
    ip = slice(1, ny)
    jp = slice(1, nx)

    phi_state = st.PHI[ip, jp]
    solidP = 1.0 - phi_state
    dryP = phi_state < cfg.phi_dry_crit
    fullP = phi_state >= cfg.phi_full_crit

    SXY_P = basic_to_p_average(st.SXY, cfg)
    COH_P = basic_to_p_average(st.COH, cfg)
    TEN_P = basic_to_p_average(st.TEN, cfg)
    FRI_P = basic_to_p_average(st.FRI, cfg)
    tauII_solid_P = np.sqrt(
        0.5 * (st.SXX[ip, jp] * st.SXX[ip, jp] + st.SYY[ip, jp] * st.SYY[ip, jp])
        + SXY_P[ip, jp] * SXY_P[ip, jp]
    )
    tauII_total_P = solidP * tauII_solid_P
    Py = tauII_total_P - TEN_P[ip, jp]

    Pc = st.pr[ip, jp] - st.pf[ip, jp]

    # Split the Keller yield envelope by branch.  Compaction viscosity XI is
    # weakened only on the tensile/Griffith branch, Pe <= Pe*.  For connected
    # melt nodes Pe = Pc; dry/under-connected nodes are excluded below.
    sin_phi = np.clip(FRI_P[ip, jp], 0.0, 1.0 - 1.0e-12)
    cos_phi = np.sqrt(np.maximum(1.0 - sin_phi * sin_phi, 0.0))
    pe_star = (COH_P[ip, jp] * cos_phi - TEN_P[ip, jp]) / np.maximum(1.0 - sin_phi, 1.0e-12)
    tensile_branch_active = (~dryP) & (~fullP) & (Pc <= pe_star)

    xi = np.maximum(st.XI[ip, jp], 1.0e-300)
    beta = st.BETTAPHI[ip, jp]
    has_elastic_storage = beta > 0.0

    Kphi = np.zeros_like(phi_state)
    Kphi[has_elastic_storage] = 1.0 / np.maximum(beta[has_elastic_storage], 1.0e-300)

    Pc_trial = np.full_like(phi_state, np.nan)
    Pc_trial[has_elastic_storage] = Pc[has_elastic_storage] * (
        Kphi[has_elastic_storage] * dt + xi[has_elastic_storage]
    ) / xi[has_elastic_storage]

    st.DSYT.fill(0.0)

    XI5 = st.XI0.copy()
    YNYT5 = np.zeros_like(st.YNYT)

    old = st.YNYT[ip, jp].copy()
    old_active = old & tensile_branch_active
    ynpl = int(np.count_nonzero(old_active))
    ddd = float(np.sum((Pc[old_active] - Py[old_active]) ** 2)) if ynpl > 0 else 0.0
    dsyt_local = st.DSYT[ip, jp]
    dsyt_local[old_active] = Pc[old_active] - Py[old_active]

    raw_active = has_elastic_storage & tensile_branch_active & (Pc_trial < Py)
    den = Pc_trial - Py
    valid = raw_active & (Py < 0.0) & (den < -1.0e-300)
    invalid_count = int(np.count_nonzero(raw_active & (~valid)))

    xipl = np.full_like(phi_state, np.inf)
    xipl[valid] = dt * Kphi[valid] * Py[valid] / den[valid]
    xipl = np.clip(xipl, cfg.xi_min, cfg.xi_max)

    base_xi = np.maximum(st.XI0[ip, jp], 1.0e-300)
    new = valid & np.isfinite(xipl) & (xipl < base_xi)
    if np.any(new):
        etalocal = XI5[ip, jp]
        etalocal[new] = (xipl[new] ** (1.0 - cfg.etawt)) * (xi[new] ** cfg.etawt)
        etalocal[new] = np.clip(etalocal[new], cfg.xi_min, cfg.xi_max)
        XI5[ip, jp] = etalocal

        ylocal = YNYT5[ip, jp]
        ylocal[new] = True
        YNYT5[ip, jp] = ylocal

        add = new & (~old)
        if np.any(add):
            dsyt_local[add] = Pc[add] - Py[add]
            ddd += float(np.sum(dsyt_local[add] ** 2))
            ynpl += int(np.count_nonzero(add))

    st.DSYT[ip, jp] = dsyt_local
    copy_pnode_edges(st.DSYT)
    copy_pnode_edges(XI5)
    copy_pnode_edges(YNYT5)

    yerr = math.sqrt(ddd / ynpl) if ynpl > 0 else 0.0
    return TensilePlasticResult(
        XI5=XI5,
        YNYT5=YNYT5,
        yerr=yerr,
        ynpl=ynpl,
        ynyt_old_count=int(np.count_nonzero(st.YNYT)),
        ynyt5_count=int(np.count_nonzero(YNYT5)),
        ynyt_changed_count=int(np.count_nonzero(st.YNYT != YNYT5)),
        invalid_count=invalid_count,
    )


def compute_aphi(st: State, cfg: Config, dt_for_elastic: float) -> None:
    ny, nx = cfg.ny, cfg.nx
    ip = slice(1, ny)
    jp = slice(1, nx)
    phi_state, phi_fluid_mat, phi_solid_mat = material_phase_fractions(st.PHI[ip, jp], cfg)
    st.APHI.fill(0.0)
    # B-style time-centred compaction rate: average old/new compaction pressure
    # in the viscous term, while the elastic-storage term uses the centred
    # pressure difference over the accepted physical timestep.
    pc_new = st.pr[ip, jp] - st.pf[ip, jp]
    pc_old = st.pr0[ip, jp] - st.pf0[ip, jp]
    st.APHI[ip, jp] = (
        0.5 * (pc_new + pc_old) / np.maximum(st.XI[ip, jp], 1.0e-300)
        + (pc_new - pc_old) / dt_for_elastic * st.BETTAPHI[ip, jp]
    ) / (phi_solid_mat * phi_fluid_mat)
    endpointP = (phi_state < cfg.phi_dry_crit) | (phi_state >= cfg.phi_full_crit)
    aphi_local = st.APHI[ip, jp]
    aphi_local[endpointP] = 0.0
    st.APHI[ip, jp] = aphi_local
    copy_pnode_edges(st.APHI)


def compute_pressure_node_velocities(st: State, cfg: Config) -> None:
    ny, nx = cfg.ny, cfg.nx
    vxleft = -cfg.strainrate * cfg.xsize / 2.0
    vxright = cfg.strainrate * cfg.xsize / 2.0
    vytop = cfg.strainrate * cfg.ysize
    vybottom = 0.0

    st.vxp.fill(0.0)
    st.vyp.fill(0.0)
    st.vxp[1:ny, 1:nx] = 0.5 * (st.vx[1:ny, 1:nx] + st.vx[1:ny, 0:nx - 1])
    st.vyp[1:ny, 1:nx] = 0.5 * (st.vy[1:ny, 1:nx] + st.vy[0:ny - 1, 1:nx])
    # Free slip sides plus imposed normal velocity extensions, matching MATLAB blocks.
    st.vxp[0, 1:nx - 1] = st.vxp[1, 1:nx - 1]
    st.vxp[ny, 1:nx - 1] = st.vxp[ny - 1, 1:nx - 1]
    st.vxp[:, 0] = 2.0 * vxleft - st.vxp[:, 1]
    st.vxp[:, nx] = 2.0 * vxright - st.vxp[:, nx - 1]
    st.vyp[1:ny - 1, 0] = st.vyp[1:ny - 1, 1]
    st.vyp[1:ny - 1, nx] = st.vyp[1:ny - 1, nx - 1]
    # Use the actual solved top-face velocity for the marker-advection ghost row.
    st.vyp[0, :] = 2.0 * st.vy[0, :] - st.vyp[1, :]
    st.vyp[ny, :] = 2.0 * vybottom - st.vyp[ny - 1, :]


def marker_timestep(st: State, cfg: Config, dt: float) -> float:
    dtm = dt
    maxvx = float(np.nanmax(np.abs(st.vx)))
    maxvy = float(np.nanmax(np.abs(st.vy)))
    if maxvx > 0.0:
        dtm = min(dtm, cfg.dxymax * cfg.dx / maxvx)
    if maxvy > 0.0:
        dtm = min(dtm, cfg.dxymax * cfg.dy / maxvy)
    aphimax = float(np.nanmax(np.abs(st.APHI)))
    if aphimax > 0.0:
        dtm = min(dtm, cfg.dphimax / aphimax)
    return max(dtm, 0.0)


def update_marker_viscosity_from_nodes(st: State, cfg: Config) -> None:
    i, j, w00, w10, w01, w11 = bilinear_indices_weights(st.xm, st.ym, st.ETA.shape, cfg, x0=0.0, y0=0.0)
    y00 = st.YNY[i, j]
    y10 = st.YNY[i, j + 1]
    y01 = st.YNY[i + 1, j]
    y11 = st.YNY[i + 1, j + 1]
    e00 = st.ETA[i, j]
    e10 = st.ETA[i, j + 1]
    e01 = st.ETA[i + 1, j]
    e11 = st.ETA[i + 1, j + 1]
    denom = (
        y00.astype(float) * w00 / np.maximum(e00, 1.0e-300)
        + y10.astype(float) * w10 / np.maximum(e10, 1.0e-300)
        + y01.astype(float) * w01 / np.maximum(e01, 1.0e-300)
        + y11.astype(float) * w11 / np.maximum(e11, 1.0e-300)
    )
    has = y00 | y10 | y01 | y11
    eta_matrix = material_matrix_viscosity(st.tm, st.phim, cfg)
    new_eta = eta_matrix.copy()
    sel = has & (denom > 0.0)
    new_eta[sel] = 1.0 / denom[sel]
    over = new_eta >= eta_matrix
    new_eta[over] = eta_matrix[over]
    st.etavpm[...] = new_eta

def update_marker_xi_from_nodes(st: State, cfg: Config) -> None:
    """Update marker effective compaction viscosity from yielded P nodes.

    This mirrors update_marker_viscosity_from_nodes() for shear viscosity, but
    uses P-node tensile/compaction plastic flags YNYT and effective compaction
    viscosity XI.  If no surrounding P node is yielded, the marker value is
    reset to the unyielded xi0 material law.
    """
    i, j, w00, w10, w01, w11 = bilinear_indices_weights(
        st.xm, st.ym, st.XI.shape, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0
    )
    y00 = st.YNYT[i, j]
    y10 = st.YNYT[i, j + 1]
    y01 = st.YNYT[i + 1, j]
    y11 = st.YNYT[i + 1, j + 1]
    x00 = st.XI[i, j]
    x10 = st.XI[i, j + 1]
    x01 = st.XI[i + 1, j]
    x11 = st.XI[i + 1, j + 1]
    denom = (
        y00.astype(float) * w00 / np.maximum(x00, 1.0e-300)
        + y10.astype(float) * w10 / np.maximum(x10, 1.0e-300)
        + y01.astype(float) * w01 / np.maximum(x01, 1.0e-300)
        + y11.astype(float) * w11 / np.maximum(x11, 1.0e-300)
    )
    has = y00 | y10 | y01 | y11
    xi0_marker = material_xi0_viscosity(st.tm, st.phim, cfg)
    new_xi = xi0_marker.copy()
    sel = has & (denom > 0.0)
    new_xi[sel] = 1.0 / denom[sel]
    over = new_xi >= xi0_marker
    new_xi[over] = xi0_marker[over]
    st.xivpm[...] = new_xi



def _bounded_shift_to_mean(
    values: np.ndarray,
    target_mean: float,
    lower: float = 0.0,
    upper: float = 1.0,
) -> np.ndarray:
    """Add a uniform offset while preserving bounds and a requested mean."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    target = float(np.clip(target_mean, lower, upper))
    lo = lower - float(np.max(values))
    hi = upper - float(np.min(values))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        trial_mean = float(np.mean(np.clip(values + mid, lower, upper)))
        if trial_mean < target:
            lo = mid
        else:
            hi = mid
    return np.clip(values + 0.5 * (lo + hi), lower, upper)


def reseed_markers_keller(st: State, cfg: Config) -> tuple[int, int, int]:
    """Rebuild under/over-populated cells with a regular marker set.

    The target is ``markers_per_cell**2`` markers per cell.  Following the
    Keller Appendix-A.5 strategy, a cell is rebuilt when its current count
    differs from that target by more than 25% (configurable).  All old markers
    in a flagged cell are removed and replaced by a uniform regular set.

    Marker properties are copied from the nearest old marker in that cell.
    For an empty cell, the nearest available markers in surrounding cells are
    used as a fallback.  Porosity and the three carried stress components are
    subsequently shifted so that their pre/post-reseeding cell means agree;
    the porosity correction is bounded to [0, 1].
    """
    if not cfg.marker_reseed or st.xm.size == 0:
        return 0, int(st.xm.size), int(st.xm.size)

    ncx = cfg.nx - 1
    ncy = cfg.ny - 1
    ncell = ncx * ncy
    target_count = cfg.markers_per_cell**2
    tolerance = cfg.marker_reseed_tolerance * target_count

    cell_j = np.clip((st.xm / cfg.dx).astype(np.int64), 0, ncx - 1)
    cell_i = np.clip((st.ym / cfg.dy).astype(np.int64), 0, ncy - 1)
    cell_id = cell_i * ncx + cell_j
    counts = np.bincount(cell_id, minlength=ncell)
    flagged = np.flatnonzero(np.abs(counts - target_count) > tolerance)
    if flagged.size == 0:
        return 0, int(st.xm.size), int(st.xm.size)

    order = np.argsort(cell_id, kind="stable")
    offsets = np.empty(ncell + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    flagged_lookup = np.zeros(ncell, dtype=bool)
    flagged_lookup[flagged] = True
    keep = ~flagged_lookup[cell_id]
    old_total = int(st.xm.size)

    marker_names = (
        "xm", "ym", "tm", "phim", "etavpm", "xivpm",
        "sxxm", "syym", "sxym",
    )
    chunks: dict[str, list[np.ndarray]] = {
        name: [getattr(st, name)[keep].copy()] for name in marker_names
    }

    sub = (np.arange(cfg.markers_per_cell, dtype=np.float64) + 0.5) / cfg.markers_per_cell
    local_x, local_y = np.meshgrid(sub, sub, indexing="xy")
    local_x = local_x.ravel()
    local_y = local_y.ravel()

    def indices_in_cell(cid: int) -> np.ndarray:
        return order[offsets[cid]:offsets[cid + 1]]

    def fallback_candidates(ii: int, jj: int) -> np.ndarray:
        max_radius = max(ncx, ncy)
        for radius in range(1, max_radius + 1):
            i0 = max(0, ii - radius)
            i1 = min(ncy - 1, ii + radius)
            j0 = max(0, jj - radius)
            j1 = min(ncx - 1, jj + radius)
            candidates: list[np.ndarray] = []
            for ni in range(i0, i1 + 1):
                for nj in range(j0, j1 + 1):
                    if ni not in (i0, i1) and nj not in (j0, j1):
                        continue
                    idx = indices_in_cell(ni * ncx + nj)
                    if idx.size:
                        candidates.append(idx)
            if candidates:
                return np.concatenate(candidates)
        return np.arange(st.xm.size, dtype=np.int64)

    for cid in flagged:
        ii = int(cid // ncx)
        jj = int(cid % ncx)
        old_idx = indices_in_cell(int(cid))
        source_idx = old_idx if old_idx.size else fallback_candidates(ii, jj)
        if source_idx.size == 0:
            raise RuntimeError(f"No source marker found while reseeding cell ({ii}, {jj}).")

        new_x = (jj + local_x) * cfg.dx
        new_y = (ii + local_y) * cfg.dy
        np.clip(new_x, 1.0e-9, cfg.xsize - 1.0e-9, out=new_x)
        np.clip(new_y, 1.0e-9, cfg.ysize - 1.0e-9, out=new_y)

        dist2 = (
            (new_x[:, None] - st.xm[source_idx][None, :])**2
            + (new_y[:, None] - st.ym[source_idx][None, :])**2
        )
        nearest = source_idx[np.argmin(dist2, axis=1)]

        new_values = {
            "xm": new_x,
            "ym": new_y,
            "tm": st.tm[nearest].copy(),
            "phim": st.phim[nearest].copy(),
            "etavpm": st.etavpm[nearest].copy(),
            "xivpm": st.xivpm[nearest].copy(),
            "sxxm": st.sxxm[nearest].copy(),
            "syym": st.syym[nearest].copy(),
            "sxym": st.sxym[nearest].copy(),
        }

        if old_idx.size:
            new_values["phim"] = _bounded_shift_to_mean(
                new_values["phim"], float(np.mean(st.phim[old_idx]))
            )
            for name in ("sxxm", "syym", "sxym"):
                new_values[name] += (
                    float(np.mean(getattr(st, name)[old_idx]))
                    - float(np.mean(new_values[name]))
                )

        for name in marker_names:
            chunks[name].append(np.asarray(new_values[name], dtype=getattr(st, name).dtype))

    for name in marker_names:
        setattr(st, name, np.concatenate(chunks[name]))

    # Keep carried viscosity histories positive and no weaker/stronger than the
    # corresponding current unyielded marker material laws permit.
    eta0_marker = material_matrix_viscosity(st.tm, st.phim, cfg)
    xi0_marker = material_xi0_viscosity(st.tm, st.phim, cfg)
    st.etavpm = np.minimum(
        np.clip(st.etavpm, cfg.eta_min, cfg.eta_max), eta0_marker
    )
    st.xivpm = np.minimum(
        np.clip(st.xivpm, cfg.xi_min, cfg.xi_max), xi0_marker
    )

    full_melt = st.phim >= cfg.phi_full_crit
    st.sxxm[full_melt] = 0.0
    st.syym[full_melt] = 0.0
    st.sxym[full_melt] = 0.0

    return int(flagged.size), old_total, int(st.xm.size)


def update_marker_stress_phi_and_advect(st: State, cfg: Config, dt: float) -> float:
    # Marker update uses the same accepted physical timestep as the mechanical solve.
    dtm = dt
    compute_aphi(st, cfg, dtm)

    # Interpolate stress increments to markers.
    dsxx = st.SXX - st.SXX0
    dsyy = st.SYY - st.SYY0
    dsxy = st.SXY - st.SXY0
    st.sxxm += interp_from_grid(dsxx, st.xm, st.ym, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    st.syym += interp_from_grid(dsyy, st.xm, st.ym, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    st.sxym += interp_from_grid(dsxy, st.xm, st.ym, cfg, x0=0.0, y0=0.0)

    # B-style porosity update: evolve phi on its native P grid first, then
    # interpolate only the nodal increment to rock markers.
    rocks = st.tm < 3
    if np.any(rocks):
        phi_grid_old = np.clip(st.PHI, 0.0, 1.0)
        factor_grid = np.exp(np.clip(st.APHI * dtm, -50.0, 50.0))
        phi_grid_new = phi_grid_old / ((1.0 - phi_grid_old) * factor_grid + phi_grid_old)
        dphi_grid = phi_grid_new - phi_grid_old
        dphim = interp_from_grid(
            dphi_grid,
            st.xm[rocks],
            st.ym[rocks],
            cfg,
            x0=-cfg.dx / 2.0,
            y0=-cfg.dy / 2.0,
        )
        st.phim[rocks] = np.clip(st.phim[rocks] + dphim, 0.0, 1.0)

        recharge_source_markers(st, cfg)

    # Basic-node rotation rate.
    st.wyx[:, :] = 0.5 * ((st.vy[0:cfg.ny, 1:cfg.nx + 1] - st.vy[0:cfg.ny, 0:cfg.nx]) / cfg.dx - (st.vx[1:cfg.ny + 1, 0:cfg.nx] - st.vx[0:cfg.ny, 0:cfg.nx]) / cfg.dy)
    omega = interp_from_grid(st.wyx, st.xm, st.ym, cfg, x0=0.0, y0=0.0)
    theta = dtm * omega
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    sxx_old = st.sxxm.copy()
    syy_old = st.syym.copy()
    sxy_old = st.sxym.copy()
    mean_n = 0.5 * (sxx_old + syy_old)
    diff_n = 0.5 * (sxx_old - syy_old)
    st.sxxm[...] = mean_n + diff_n * c2 - sxy_old * s2
    st.syym[...] = mean_n - diff_n * c2 + sxy_old * s2
    st.sxym[...] = diff_n * s2 + sxy_old * c2

    # True full-melt markers have no solid-skeleton stress history.
    full_melt_markers = st.phim >= cfg.phi_full_crit
    st.sxxm[full_melt_markers] = 0.0
    st.syym[full_melt_markers] = 0.0
    st.sxym[full_melt_markers] = 0.0

    # RK4 marker advection.
    if dtm > 0.0:
        compute_pressure_node_velocities(st, cfg)
        x0 = st.xm.copy()
        y0 = st.ym.copy()
        k1x, k1y = solid_velocity_at(st, cfg, x0, y0)
        k2x, k2y = solid_velocity_at(st, cfg, x0 + 0.5 * dtm * k1x, y0 + 0.5 * dtm * k1y)
        k3x, k3y = solid_velocity_at(st, cfg, x0 + 0.5 * dtm * k2x, y0 + 0.5 * dtm * k2y)
        k4x, k4y = solid_velocity_at(st, cfg, x0 + dtm * k3x, y0 + dtm * k3y)
        st.xm[...] = x0 + dtm / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        st.ym[...] = y0 + dtm / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        np.clip(st.xm, 1.0e-9, cfg.xsize - 1.0e-9, out=st.xm)
        np.clip(st.ym, 1.0e-9, cfg.ysize - 1.0e-9, out=st.ym)

    recharge_source_markers(st, cfg)

    # Keller-style marker reseeding is performed only after the physical step
    # has been accepted and marker advection is complete.  Therefore timestep
    # retries always restore a state with an unchanged marker-array shape.
    reseed_markers_keller(st, cfg)

    # Reproject after marker update for plotting and for the next step's state.
    interpolate_markers_to_nodes(st, cfg, with_stress=True)
    return dtm


def solid_velocity_at(st: State, cfg: Config, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx_p = interp_from_grid(st.vxp, x, y, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    vy_p = interp_from_grid(st.vyp, x, y, cfg, x0=-cfg.dx / 2.0, y0=-cfg.dy / 2.0)
    vx_s = interp_from_grid(st.vx, x, y, cfg, x0=0.0, y0=-cfg.dy / 2.0, max_i0=cfg.ny - 1, max_j0=cfg.nx - 2)
    vy_s = interp_from_grid(st.vy, x, y, cfg, x0=-cfg.dx / 2.0, y0=0.0, max_i0=cfg.ny - 2, max_j0=cfg.nx - 1)
    return cfg.vpratio * vx_p + (1.0 - cfg.vpratio) * vx_s, cfg.vpratio * vy_p + (1.0 - cfg.vpratio) * vy_s


def advance_one_step(
    st: State,
    cfg: Config,
    it: int,
    dt: float,
    Vpdt: float,
    Kfdt: float,
    rhof_dt: float,
    *,
    runtime_logger: RuntimeLogger | None = None,
) -> StepStats:
    step_start = snapshot_state(st) # 保留全部状态量
    retries = 0
    dt_current = dt

    while True:
        restore_state(st, step_start) # 把前面保存的状态量 赋给状态数组
        interpolate_markers_to_nodes(st, cfg, with_stress=True)
        apply_velocity_bc(st, cfg)
        apply_hydraulic_bc(st, cfg)
        enforce_darcy_flux_bc(st, cfg)
        st.pr0[...] = st.pr
        st.pf0[...] = st.pf
        st.PHI0[...] = st.PHI
        st.SXX0[...] = st.SXX
        st.SYY0[...] = st.SYY
        st.SXY0[...] = st.SXY
        apply_low_phi_pressure_projection(st, cfg, include_history=True)
        apply_low_phi_darcy_cutoff(st, cfg)

        # BETTAPHI has already been updated inside interpolate_markers_to_nodes().
        # Preserve the original startup treatment: at it == 0 the pore-elastic
        # storage term is disabled for the first solve.
        if it == 0:
            st.BETTAPHI.fill(0.0)
            copy_pnode_edges(st.BETTAPHI)

        ETA00 = st.ETA.copy()
        YNY00 = st.YNY.copy()
        XI00 = st.XI.copy()
        YNYT00 = st.YNYT.copy()

        total_pt = 0
        resid = 2.0 * cfg.epsi
        err_vy = 0.0
        iplast_done = 0
        pt_all = True
        pr = PlasticResult(st.ETA.copy(), st.YNY.copy(), 0.0, 0, int(np.count_nonzero(st.YNY)), int(np.count_nonzero(st.YNY)), 0)
        tr = TensilePlasticResult(st.XI.copy(), st.YNYT.copy(), 0.0, 0, int(np.count_nonzero(st.YNYT)), int(np.count_nonzero(st.YNYT)), 0, 0)
        converged = False

        for iplast in range(1, cfg.nplast + 1):
            iplast_done = iplast
            niter, resid, err_vy, pt_conv = solve_hm_fixed_eta(
                st,
                cfg,
                it,
                dt_current,
                Vpdt,
                Kfdt,
                rhof_dt,
                iplast=iplast,
                runtime_logger=runtime_logger,
            )
            total_pt += niter
            pt_all = pt_all and pt_conv
            pr = compute_plastic_active_set(st, cfg, dt_current)
            tr = compute_tensile_plastic_active_set(st, cfg, dt_current)
            shear_converged = (pr.ynpl == 0) or (pr.yerr < cfg.yerrmax)
            tensile_converged = (tr.ynpl == 0) or (tr.yerr < cfg.yerrmax)
            converged = shear_converged and tensile_converged
            if runtime_logger is not None:
                runtime_logger.log_plastic(
                    st,
                    cfg,
                    pr,
                    tr,
                    step=it,
                    iplast=iplast,
                    dt=dt_current,
                    pt_iters_this_plastic=niter,
                    pt_resid=resid,
                    pt_converged=pt_conv,
                    shear_converged=shear_converged,
                    tensile_converged=tensile_converged,
                    plastic_converged=converged,
                    retries=retries,
                )
            if converged or iplast == cfg.nplast:
                break
            if cfg.dtstep > 0 and iplast % cfg.dtstep == 0:
                retries += 1
                if retries > cfg.max_dt_retries:
                    break
                dt_current /= cfg.dtkoef
                set_eta_yny(st, ETA00, YNY00)
                set_tensile_yny(st, XI00, YNYT00)
                compute_etap(st, cfg)
                continue
            set_eta_yny(st, pr.ETA5, pr.YNY5)
            set_tensile_yny(st, tr.XI5, tr.YNYT5)
            compute_etap(st, cfg)


        # The marker CFL limit constrains the complete physical timestep.
        # Do not solve mechanics with dt_current and then advance porosity,
        # stress history, and markers with a smaller dtm.  If the stable marker
        # timestep is smaller, restore the beginning-of-step state through the
        # outer retry loop and recompute every equation with the reduced dt.
        compute_aphi(st, cfg, dt_current)
        dt_stable = marker_timestep(st, cfg, dt_current)
        if dt_stable < dt_current * (1.0 - 1.0e-10):
            retries += 1
            if retries > cfg.max_dt_retries or dt_stable <= 0.0:
                restore_state(st, step_start)
                raise RuntimeError(
                    f"marker CFL failed at step {it + 1}; "
                    f"dt={dt_current:.6e}, stable dt={dt_stable:.6e}"
                )
            # A small safety margin prevents round-off from repeatedly
            # triggering the same limiter after the nonlinear fields are
            # recomputed at the reduced timestep.
            dt_current = 0.95 * dt_stable
            continue

        update_keller_step1_diagnostics(st, cfg)
        update_marker_viscosity_from_nodes(st, cfg)
        update_marker_xi_from_nodes(st, cfg)
        dtm = update_marker_stress_phi_and_advect(st, cfg, dt_current)
        return StepStats(
            pt_iters=total_pt,
            resid=resid,
            err_vy=err_vy,
            dt=dt_current,
            dtm=dtm,
            iplast=iplast_done,
            yerr=pr.yerr,
            ynpl=pr.ynpl,
            yny_count=int(np.count_nonzero(st.YNY)),
            yny5_count=pr.yny5_count,
            yny_changed_count=pr.yny_changed_count,
            tyerr=tr.yerr,
            tynpl=tr.ynpl,
            tyny_count=int(np.count_nonzero(st.YNYT)),
            tyny5_count=tr.ynyt5_count,
            tyny_changed_count=tr.ynyt_changed_count,
            tinvalid_count=tr.invalid_count,
            retries=retries,
            converged=converged,
            pt_converged=pt_all,
        )


def marker_count_diagnostics(st: State, cfg: Config) -> tuple[np.ndarray, np.ndarray, float]:
    """Return marker-count heatmaps on the basic-node plotting grid.

    Counts are accumulated per grid cell and then padded to (Ny, Nx) so they
    can be plotted on the same basic-node mesh as the other diagnostic panels.
    The second returned array is a 0/1 mask for cells whose count is below
    25% of the current domain-average marker count.
    """
    ix = np.clip((st.xm / cfg.dx).astype(np.int64), 0, cfg.nx - 2)
    iy = np.clip((st.ym / cfg.dy).astype(np.int64), 0, cfg.ny - 2)

    counts_cell = np.zeros((cfg.ny - 1, cfg.nx - 1), dtype=np.int32)
    np.add.at(counts_cell, (iy, ix), 1)

    mean_count = float(np.mean(counts_cell))
    low_mask_cell = counts_cell < (0.25 * mean_count)

    counts_plot = np.pad(counts_cell, ((0, 1), (0, 1)), mode="edge").astype(np.float64)
    low_mask_plot = np.pad(low_mask_cell, ((0, 1), (0, 1)), mode="edge").astype(np.float64)
    return counts_plot, low_mask_plot, mean_count


def plot_variables(st: State, cfg: Config) -> dict[str, np.ndarray]:
    """Return the exact fields used by the multi-panel diagnostic plot.

    All returned 2-D arrays are on the basic-node plotting grid (Ny, Nx), so
    the PNG panels and .mat files contain matching fields.
    """
    eps_bg = max(abs(cfg.strainrate), 1.0e-300)
    marker_count, marker_low25_mask, marker_mean_count = marker_count_diagnostics(st, cfg)
    return {
        "phi_pct": 100.0 * p_to_basic_average(st.PHI, cfg),
        "log10_eta": np.log10(np.maximum(st.ETA, 1.0e-300)),
        "divv_scaled": p_to_basic_average(st.DIVV, cfg) / eps_bg,
        "Pc_MPa": p_to_basic_average(st.pr - st.pf, cfg) / 1.0e6,
        "Pf_MPa": p_to_basic_average(st.pf, cfg) / 1.0e6,
        "Pt_MPa": p_to_basic_average(st.pr, cfg) / 1.0e6,
        "qyD_m_per_s": p_to_basic_average(st.qyD, cfg),
        "qxD_m_per_s": p_to_basic_average(st.qxD, cfg),
        "marker_count": marker_count,
        "marker_low25_mask": marker_low25_mask,
        "marker_mean_count": np.array([[marker_mean_count]], dtype=np.float64),
    }


def save_plot_variables_mat(st: State, cfg: Config, frame: int, timestep: int, outdir: Path, model_time_s: float) -> Path:
    """Save the plotted variables to a MATLAB .mat file."""
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        from scipy.io import savemat
    except Exception as exc:
        raise RuntimeError("Saving .mat files requires scipy; install scipy or disable MAT saving.") from exc

    c = coords(cfg)
    fields = plot_variables(st, cfg)
    mat = {
        "x_km": c["x"] / 1000.0,
        "y_km": c["y"] / 1000.0,
        "frame": np.array([[frame]], dtype=np.int64),
        "timestep": np.array([[timestep]], dtype=np.int64),
        "time_s": np.array([[model_time_s]], dtype=np.float64),
        "time_kyr": np.array([[model_time_s / SECONDS_PER_KYR]], dtype=np.float64),
        "source_pressure_Pa": np.array([[cfg.source_pressure]], dtype=np.float64),
        "source_density_kg_m3": np.array([[cfg.source_density]], dtype=np.float64),
    }
    mat.update(fields)
    path = outdir / f"plotvars_timestep_{timestep:04d}_frame_{frame:03d}.mat"
    savemat(path, mat, do_compression=True)
    return path


def plot_state(st: State, cfg: Config, frame: int, outdir: Path, model_time_s: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    c = coords(cfg)
    X, Y = np.meshgrid(c["x"] / 1000.0, c["y"] / 1000.0, indexing="xy")
    f = plot_variables(st, cfg)

    fig, axs = plt.subplots(2, 5, figsize=(24.0, 8.8), constrained_layout=True)
    fig.suptitle(
        f"Keller d118r2-style run; t = {model_time_s / SECONDS_PER_KYR:.2f} kyr; frame = {frame}",
        fontsize=14,
        fontweight="bold",
    )

    qmax = max(
        float(np.nanmax(np.abs(f["qxD_m_per_s"]))),
        float(np.nanmax(np.abs(f["qyD_m_per_s"]))),
        1.0e-30,
    )
    divmax = max(float(np.nanmax(np.abs(f["divv_scaled"]))), 1.0e-12)
    marker_count_max = max(float(np.nanmax(f["marker_count"])), 1.0)
    phi_max_pct = float(np.nanmax(f["phi_pct"]))

    panels = [
        (axs[0, 0], f["phi_pct"], f"Melt fraction / porosity [%]\nMax porosity = {phi_max_pct:.2f}%", "magma", 0.0, 100.0),
        (axs[0, 1], f["log10_eta"], r"log$_{10}$ viscosity [Pa s]", "viridis", 16.0, 18.1),
        (axs[0, 2], f["divv_scaled"], r"Volumetric strain rate $\nabla\cdot v_s/|\dot\epsilon_{BG}|$", "seismic", -divmax, divmax),
        (axs[0, 3], f["Pc_MPa"], r"Compaction pressure $P_c=P_t-P_f$ [MPa]", "seismic", -5.0, 5.0),
        (axs[0, 4], f["marker_count"], "Marker count per cell", "viridis", 0.0, marker_count_max),
        (axs[1, 0], f["Pf_MPa"], r"Fluid pressure $P_f$ [MPa]", "viridis", None, None),
        (axs[1, 1], f["Pt_MPa"], r"Total pressure $P_t=P_r$ [MPa]", "viridis", None, None),
        (axs[1, 2], f["qyD_m_per_s"], r"Darcy flux $q_y^D$ [m/s]", "seismic", -qmax, qmax),
        (axs[1, 3], f["qxD_m_per_s"], r"Darcy flux $q_x^D$ [m/s]", "seismic", -qmax, qmax),
        (axs[1, 4], f["marker_low25_mask"], "Marker count < 25% of mean", "gray_r", 0.0, 1.0),
    ]
    for ax, arr, title, cmap, vmin, vmax in panels:
        im = ax.pcolormesh(X, Y, arr, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Width [km]")
        ax.set_ylabel("Depth [km]")
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()
        if cfg.sticky_air_thickness > 0.0:
            ax.axhline(
                cfg.sticky_air_thickness / 1000.0,
                color="white",
                linewidth=0.8,
                linestyle="--",
                alpha=0.8,
            )
        fig.colorbar(im, ax=ax)

    path = outdir / f"frame_{frame:03d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_eta_only(st: State, cfg: Config, timestep: int, outdir: Path, model_time_s: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(np.log10(np.maximum(st.ETA, 1.0e-300)), origin="upper", cmap="jet", vmin=17, vmax=23)
    ax.set_title(f"log10ETA, Pa*s timestep={timestep}; t={model_time_s / SECONDS_PER_KYR:.3f} kyr")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax)
    path = outdir / f"eta_timestep_{timestep:04d}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def make_gif(outdir: Path, gif_name: str = "animation.gif", fps: float = 6.0) -> Path | None:
    try:
        import imageio.v2 as imageio
    except Exception:
        return None

    def key(path: Path) -> int:
        m = re.search(r"frame_(\d+)\.png$", path.name)
        return int(m.group(1)) if m else 10**12

    frames = sorted(outdir.glob("frame_*.png"), key=key)
    if not frames:
        return None
    gif_path = outdir / gif_name
    with imageio.get_writer(gif_path, mode="I", duration=1.0 / max(fps, 1.0e-12)) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    return gif_path


def parse_frames(text: str) -> set[int]:
    out: set[int] = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    return out


def run(
    cfg: Config,
    outdir: Path,
    *,
    make_png: bool,
    make_animation: bool,
    plot_frames: Iterable[int],
    eta_only: bool,
    quiet: bool,
    runtime_log: bool,
    pt_log_every: int,
    pt_console_every: int,
    plastic_console_every: int,
    mat_save_every: int,
) -> None:
    validate_config(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    runtime_logger = RuntimeLogger(
        outdir,
        enabled=runtime_log,
        console=not quiet,
        pt_log_every=pt_log_every,
        pt_console_every=pt_console_every,
        plastic_console_every=plastic_console_every,
    )
    st = initial_state(cfg)
    dt = cfg.dt0
    Vpdt = cfg.CFL * cfg.dx
    vpdt_hydro = cfg.dx / 6.1
    Re2 = 0.25 * (
        PI
        + math.sqrt(
            PI * PI
            + (
                cfg.ysize
                / math.sqrt(
                    (cfg.kphi_pt_scale / cfg.etafluid)
                    * cfg.xi_pt_scale
                )
            ) ** 2
        )
    )
    rhof_dt = Re2 * cfg.xi_pt_scale / (cfg.ysize * vpdt_hydro)
    Kfdt = vpdt_hydro * Re2 * cfg.xi_pt_scale / cfg.xsize
    wanted = set(plot_frames)
    isave = 0
    model_time_s = 0.0
    t0 = time.time()
    if not quiet:
        print(
            f"MATLAB-staggered P-T marker run: nx={cfg.nx}, ny={cfg.ny}, markers={st.xm.size}, "
            f"niter={cfg.niter}, nplast={cfg.nplast}, yerrmax={cfg.yerrmax:.3e} Pa"
        )
        if runtime_log:
            print(f"Runtime P-T CSV: {runtime_logger.pt_path}")
            print(f"Runtime plastic CSV: {runtime_logger.plastic_path}")

    for it in range(cfg.nt):
        dt = min(dt * cfg.dtkoefup, cfg.dt0)
        stats = advance_one_step(st, cfg, it, dt, Vpdt, Kfdt, rhof_dt, runtime_logger=runtime_logger)
        dt = stats.dt
        model_time_s += stats.dt
        if not np.isfinite(stats.resid):
            print(f"ERROR: non-finite residual at it={it}")
            break
        if mat_save_every > 0 and it % mat_save_every == 0:
            save_plot_variables_mat(st, cfg, isave, it + 1, outdir, model_time_s)

        if it % cfg.nsave == 0:
            wrote = False
            if make_png and (not wanted or isave in wanted or (it + 1) in wanted):
                if eta_only:
                    plot_eta_only(st, cfg, it + 1, outdir, model_time_s)
                else:
                    plot_state(st, cfg, isave, outdir, model_time_s)
                wrote = True
            if not quiet:
                valid = st.SIIB > 1.0e-30
                plast = np.nanmax(1.0 - st.SYIELD[valid] / st.SIIB[valid]) if np.any(valid) else float("nan")
                ratio = st.ETA / np.maximum(st.ETA0, 1.0e-300)
                print(
                    f"{'saved' if wrote else 'skipped'} frame {isave:03d} at it={it:03d}; "
                    f"iplast={stats.iplast}; pt_iter={stats.pt_iters}; resid={stats.resid:.3e}; "
                    f"YERR={stats.yerr:.3e} Pa; ynpl={stats.ynpl}; "
                    f"YNY={stats.yny_count}; YNY5={stats.yny5_count}; dYNY={stats.yny_changed_count}; "
                    f"TYERR={stats.tyerr:.3e} Pa; tynpl={stats.tynpl}; "
                    f"YNYT={stats.tyny_count}; YNYT5={stats.tyny5_count}; dYNYT={stats.tyny_changed_count}; Tinv={stats.tinvalid_count}; "
                    f"ETA/ETA0<0.99={int(np.count_nonzero(ratio < 0.99))}; "
                    f"max Plast={plast:.3e}; mean phi={np.mean(st.PHI[1:cfg.ny,1:cfg.nx]):.6e}; "
                    f"Pc_max={np.nanmax(np.abs(st.Pc[1:cfg.ny,1:cfg.nx])):.3e}; "
                    f"BETTAPHI_max={np.nanmax(st.BETTAPHI[1:cfg.ny,1:cfg.nx]):.3e}; "
                    f"solidP_min={np.nanmin(st.solidP[1:cfg.ny,1:cfg.nx]):.3e}; "
                    f"solidB_min={np.nanmin(st.solidB):.3e}; "
                    f"dt={stats.dt:.3e}; dtm={stats.dtm:.3e}; retries={stats.retries}; "
                    f"conv={stats.converged}; pt_conv={stats.pt_converged}"
                )
            isave += 1
        elif it % cfg.nsave == 1 and it > 0 and not quiet:
            print(
                f"> it={it:05d} > iplast={stats.iplast:04d} > pt_iter={stats.pt_iters:05d} "
                f"||resid||={stats.resid:.6e}, YERR={stats.yerr:.3e} Pa, TYERR={stats.tyerr:.3e} Pa, ||Vy_MAX||={stats.err_vy:.3e}",
                flush=True,
            )

    gif_path = make_gif(outdir) if make_png and make_animation and not eta_only else None
    runtime_logger.close()
    if not quiet:
        print(f"Done: {it + 1} steps in {time.time() - t0:.2f} s. Output: {outdir}")
        if gif_path:
            print(f"GIF: {gif_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Keller et al. (2013) d118r2-like VEP/HM marker setup with pure lower Pf source BC.")
    p.add_argument("cmd", nargs="?", choices=["run", "smoke"], default="run")
    p.add_argument("--out", type=Path, default=Path("keller_d118r2_integrated_pressure_sticky_air_output"))
    p.add_argument("--nx", type=int, default=181)
    p.add_argument("--ny", type=int, default=121, help="Number of nodes through the 4 km rock column; sticky-air nodes are added automatically.")
    p.add_argument("--nt", type=int, default=1500)
    p.add_argument("--niter", type=int, default=8000)
    p.add_argument("--epsi", type=float, default=1.0e-7)
    p.add_argument("--nout", type=int, default=50)
    p.add_argument("--nsave", type=int, default=5)
    p.add_argument("--xsize", type=float, default=6_000.0)
    p.add_argument("--ysize", type=float, default=4_000.0, help="Rock-column thickness [m], excluding sticky air.")
    p.add_argument("--sticky-air-thickness", type=float, default=500.0, help="Top sticky-air thickness [m].")
    p.add_argument("--eta-air", type=float, default=1.0e16, help="Sticky-air viscosity [Pa s].")
    p.add_argument("--rho-air", type=float, default=1.0, help="Sticky-air density [kg/m^3].")
    p.add_argument("--strainrate", type=float, default=1.0e-15)
    p.add_argument("--markers-per-cell", type=int, default=4)
    p.add_argument("--marker-seed", type=int, default=1)
    p.add_argument("--no-marker-reseed", action="store_true", help="Disable Keller-style marker reseeding; reseeding is enabled by default.")
    p.add_argument("--marker-reseed-tolerance", type=float, default=0.25, help="Fractional marker-count deviation that triggers cell rebuilding; Keller uses 0.25.")
    p.add_argument("--nplast", type=int, default=200)
    p.add_argument("--yerrmax", type=float, default=3.0e5)
    p.add_argument("--dtstep", type=int, default=200)
    p.add_argument("--dtkoef", type=float, default=1.2)
    p.add_argument("--dtkoefup", type=float, default=1.2)
    p.add_argument("--dxymax", type=float, default=0.5)
    p.add_argument("--vpratio", type=float, default=1.0 / 3.0)
    p.add_argument("--phi-pt-scale", type=float, default=0.01, help="Numerical melt-fraction scale used only by the legacy P-T fluid-pressure/Darcy relaxation.")
    p.add_argument("--phimin", type=float, default=1.0e-6, help="Lower melt fraction used only in material-property regularization; physical phi may be zero.")
    p.add_argument("--solid-fraction-min", type=float, default=1.0e-2, help="Lower solid fraction used only in material-property laws; physical phi may reach one.")
    p.add_argument("--phi-background", type=float, default=0.0, help="Background melt fraction before numerical clipping.")
    p.add_argument("--phi-amplitude", type=float, default=0.20, help="Peak amplitude of the initial Gaussian melt pulse.")
    p.add_argument("--gaussian-sigma-x", type=float, default=300.0, help="Horizontal Gaussian width scale [m].")
    p.add_argument("--gaussian-sigma-y", type=float, default=300.0, help="Vertical Gaussian width scale [m].")
    p.add_argument("--source-halfwidth", type=float, default=650.0, help="Half-width of lower pressure-reservoir melt source [m].")
    p.add_argument("--source-flux-limit", type=float, default=5.0e-10, help="Backward-compatible option; ignored by the pure Pf source-boundary version.")
    p.add_argument("--surface-pressure", type=float, default=50.0e6, help="Surface pressure/overburden offset [Pa]; Keller Section 3.1 uses 50 MPa.")
    p.add_argument("--eta0", type=float, default=1.0e18, help="Keller intrinsic host-rock viscosity eta0 [Pa s].")
    p.add_argument("--eta-min", type=float, default=1.0e16, help="Minimum effective shear viscosity [Pa s].")
    p.add_argument("--eta-max", type=float, default=1.0e23, help="Maximum effective shear viscosity [Pa s].")
    p.add_argument("--eta-fluid", type=float, default=10.0, help="Melt/fluid viscosity [Pa s].")
    p.add_argument("--alpha-phi", type=float, default=27.0, help="Keller melt-weakening factor alpha_phi.")
    p.add_argument("--cohesion", type=float, default=40.0e6, help="Cohesion C [Pa].")
    p.add_argument("--tensile-strength", type=float, default=20.0e6, help="Tensile strength sigma_T [Pa]; 20 MPa gives d118r2.")
    p.add_argument("--friction-angle-deg", type=float, default=30.0, help="Friction angle [deg]; stored internally as sin(angle).")
    p.add_argument("--kphi-elastic0", type=float, default=5.0e9, help="Keller pore modulus prefactor Kphi0 [Pa].")
    p.add_argument("--kphi-exp", type=float, default=0.5, help="Keller pore modulus exponent q in K_phi=Kphi0*phi**(-q).")
    p.add_argument("--phi-crit", type=float, default=1.0e-3, help="Under-connected melt threshold for the Keller pressure/effective-pressure switch.")
    p.add_argument("--full-melt-eps", type=float, default=1.0e-9, help="Tolerance below phi=1 used to identify the true full-melt endpoint.")
    p.add_argument("--kphi0", type=float, default=1.0e-8, help="Keller permeability prefactor k0 in k_phi=k0*phi**3*(1-phi)**2 [m^2].")
    p.add_argument("--kphi-min", type=float, default=1.0e-19, help="Lower permeability cut-off [m^2], following Keller Appendix A4 stabilization.")
    p.add_argument("--eta-melt-cutoff", type=float, default=1.0e16, help="Lower cut-off viscosity used in Keller's full-melt/high-melt total-stress limit [Pa s].")
    p.add_argument("--rho-solid", type=float, default=3000.0, help="Solid density [kg/m^3].")
    p.add_argument("--rho-fluid", type=float, default=2500.0, help="Fluid/melt density [kg/m^3].")
    p.add_argument("--gravity", type=float, default=10.0, help="Gravity magnitude [m/s^2], positive downward in the model y coordinate.")
    p.add_argument("--plot-frames", default="")
    p.add_argument("--eta-only", action="store_true")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-runtime-log", action="store_true", help="Disable detailed runtime CSV files; terminal iteration output remains active unless --quiet or console intervals are 0.")
    p.add_argument("--pt-log-every", type=int, default=1, help="Write one P-T CSV row every N P-T iterations.")
    p.add_argument("--pt-console-every", type=int, default=1, help="Print only the final P-T console line for each P-T solve; 0 disables P-T console lines.")
    p.add_argument("--plastic-console-every", type=int, default=1, help="Print one plastic console line every N plastic iterations; 0 disables plastic console lines.")
    p.add_argument("--mat-save-every", type=int, default=30, help="Save plotted variables to .mat every N time steps; 0 disables .mat output.")
    src_group = p.add_mutually_exclusive_group()
    src_group.add_argument("--source-recharge", action="store_true", help="Enable the old numerical porosity/viscosity recharge inside the lower source patch; useful only for diagnostics.")
    src_group.add_argument("--no-source-recharge", action="store_true", help="Keep source recharge disabled. This is the default in case4_notail.py.")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    rock_dy = a.ysize / max(a.ny - 1, 1)
    air_intervals = int(round(a.sticky_air_thickness / max(rock_dy, 1.0e-300)))
    total_ysize = a.ysize + a.sticky_air_thickness
    total_ny = a.ny + air_intervals

    common = dict(
        xsize=a.xsize,
        ysize=total_ysize,
        sticky_air_thickness=a.sticky_air_thickness,
        eta_air=a.eta_air,
        rho_air=a.rho_air,
        strainrate=a.strainrate,
        markers_per_cell=a.markers_per_cell,
        marker_seed=a.marker_seed,
        marker_reseed=not a.no_marker_reseed,
        marker_reseed_tolerance=a.marker_reseed_tolerance,
        epsi=a.epsi,
        phi_pt_scale=a.phi_pt_scale,
        phimin=a.phimin,
        solid_fraction_min=a.solid_fraction_min,
        phi_background=a.phi_background,
        phi_amplitude=a.phi_amplitude,
        gaussian_sigma_x=a.gaussian_sigma_x,
        gaussian_sigma_y=a.gaussian_sigma_y,
        source_halfwidth=a.source_halfwidth,
        source_flux_limit=a.source_flux_limit,
        surface_pressure=a.surface_pressure,
        eta_block=a.eta0,
        eta_weak=a.eta0,
        eta_min=a.eta_min,
        eta_max=a.eta_max,
        etafluid=a.eta_fluid,
        alphaphi=a.alpha_phi,
        coh0=a.cohesion,
        tens0=a.tensile_strength,
        fric_block=math.sin(math.radians(a.friction_angle_deg)),
        nplast=a.nplast,
        yerrmax=a.yerrmax,
        dtstep=a.dtstep,
        dtkoef=a.dtkoef,
        dtkoefup=a.dtkoefup,
        dxymax=a.dxymax,
        vpratio=a.vpratio,
        Kphi0=a.kphi_elastic0,
        Kphi_exp=a.kphi_exp,
        phi_crit=a.phi_crit,
        full_melt_eps=a.full_melt_eps,
        kphi0=a.kphi0,
        kphi_min=a.kphi_min,
        eta_melt_cutoff=a.eta_melt_cutoff,
        rho_solid=a.rho_solid,
        rho_fluid=a.rho_fluid,
        gravity=a.gravity,
        keep_source_porosity=bool(a.source_recharge and not a.no_source_recharge),
    )
    if a.cmd == "smoke":
        smoke_common = dict(common)
        smoke_common["nplast"] = min(a.nplast, 4)
        smoke_rock_ny = 31
        smoke_dy = a.ysize / max(smoke_rock_ny - 1, 1)
        smoke_air_intervals = int(round(a.sticky_air_thickness / max(smoke_dy, 1.0e-300)))
        cfg = Config(nx=31, ny=smoke_rock_ny + smoke_air_intervals, nt=3, niter=min(a.niter, 40), nout=5, nsave=1, **smoke_common)
    else:
        cfg = Config(nx=a.nx, ny=total_ny, nt=a.nt, niter=a.niter, nout=a.nout, nsave=a.nsave, **common)
    run(
        cfg,
        a.out,
        make_png=not a.no_png,
        make_animation=not a.no_gif,
        plot_frames=parse_frames(a.plot_frames),
        eta_only=a.eta_only,
        quiet=a.quiet,
        runtime_log=not a.no_runtime_log,
        pt_log_every=a.pt_log_every,
        pt_console_every=a.pt_console_every,
        plastic_console_every=a.plastic_console_every,
        mat_save_every=a.mat_save_every,
    )


if __name__ == "__main__":
    main()
