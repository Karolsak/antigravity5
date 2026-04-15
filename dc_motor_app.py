"""
dc_motor_app.py
================
Comprehensive production-quality Python Tkinter application for solving
Problems 28 and 29 from a DC Machines textbook.

Motor Specifications:
  - 4-pole, shunt-wound DC motor
  - Armature resistance Ra = 4 Ω
  - Supply voltage Vt = 240 V DC
  - No-load speed = 1800 r/min (Ia_noload ≈ 0)
  - Constant field excitation, armature reaction neglected

Run with:  python dc_motor_app.py
Dependencies: tkinter (stdlib), math (stdlib), numpy, scipy, matplotlib
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global motor constants (default values – overridden by Tab-2 sliders)
# ---------------------------------------------------------------------------
DEFAULT_VT      = 240.0   # Supply voltage [V]
DEFAULT_RA      = 4.0     # Armature resistance [Ω]
DEFAULT_N0      = 1800.0  # No-load speed [r/min]
DEFAULT_POLES   = 4       # Number of poles

# Derived constants
# K_e in V/(r/min): at no-load Vt = Ea, Ea = Ke*n  → Ke = Vt/n0
# K_e_rad in V/(rad/s) = Ke * 60/(2π)
# K_t in N·m/A = K_e_rad   (SI units, lossless)

HP_PER_WATT   = 1.0 / 745.7
FTLBF_PER_NM  = 0.737562


def clamp(val, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, val))


def ke_rad(vt, n0):
    """Back-EMF constant in V·s/rad."""
    return vt / (n0 * 2.0 * math.pi / 60.0)


def torque_speed_line(vt, ra, n0, n_array):
    """
    Compute steady-state Ia and T for a range of speeds.
    Ea = ke*(2π/60)*n,  Ia = (Vt - Ea)/Ra,  T = Kt*Ia
    """
    Ke = ke_rad(vt, n0)
    omega = n_array * (2.0 * math.pi / 60.0)
    Ea    = Ke * omega
    Ia    = (vt - Ea) / ra
    T     = Ke * Ia          # Kt = Ke in SI
    return Ia, T


# ===========================================================================
# Helper: embed a matplotlib Figure in a tk Frame
# ===========================================================================
def embed_figure(fig, parent):
    """Embed *fig* into *parent* frame; return (canvas, toolbar)."""
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas, toolbar


# ===========================================================================
# Application Class
# ===========================================================================
class DCMotorApp(tk.Tk):
    """
    Main application window containing a ttk.Notebook with 12 tabs.
    All motor parameters are stored as tk variables so tabs stay in sync.
    """

    def __init__(self):
        super().__init__()
        self.title("DC Shunt Motor Analysis – Problems 28 & 29")
        self.geometry("1300x850")
        self.minsize(900, 600)

        # ---- Shared parameter variables (updated by Tab-2 sliders) --------
        self.var_vt    = tk.DoubleVar(value=DEFAULT_VT)
        self.var_ra    = tk.DoubleVar(value=DEFAULT_RA)
        self.var_n0    = tk.DoubleVar(value=DEFAULT_N0)
        self.var_poles = tk.IntVar(value=DEFAULT_POLES)

        # ---- Simulation state flags ----------------------------------------
        self._sim_running   = False
        self._fault_running = False
        self._pid_running   = False
        self._thermal_running = False

        # ---- Build notebook ------------------------------------------------
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        self._build_tab5()
        self._build_tab6()
        self._build_tab7()
        self._build_tab8()
        self._build_tab9()
        self._build_tab10()
        self._build_tab11()
        self._build_tab12()

        # Bind resize
        self.bind("<Configure>", self._on_resize)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def _params(self):
        """Return current (Vt, Ra, n0, poles) as floats."""
        vt    = clamp(float(self.var_vt.get()),    1.0, 2000.0)
        ra    = clamp(float(self.var_ra.get()),    0.01,  200.0)
        n0    = clamp(float(self.var_n0.get()),    10.0, 20000.0)
        poles = clamp(int(self.var_poles.get()),   2,    20)
        return vt, ra, n0, poles

    def _on_resize(self, event):
        pass  # Figures auto-resize via pack fill=BOTH expand=True

    # =======================================================================
    # TAB 1 – Main Menu and Problem Solutions
    # =======================================================================
    def _build_tab1(self):
        """Tab 1: Full problem text and step-by-step solutions."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 1 – Solutions ")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Title
        ttk.Label(frame, text="DC Shunt Motor – Problems 28 & 29  (4-pole, Vt=240V, Ra=4Ω, n0=1800 r/min)",
                  font=("Helvetica", 13, "bold")).grid(row=0, column=0, pady=6, padx=8, sticky="w")

        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # ---- Left: schematic canvas ----------------------------------------
        canvas_frame = ttk.LabelFrame(paned, text="Motor Schematic")
        paned.add(canvas_frame, weight=1)
        self._tab1_canvas = tk.Canvas(canvas_frame, bg="white", width=320, height=400)
        self._tab1_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._draw_schematic(self._tab1_canvas)

        # ---- Right: solutions text ------------------------------------------
        right = ttk.Frame(paned)
        paned.add(right, weight=2)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        txt = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=("Courier", 10))
        txt.grid(row=0, column=0, sticky="nsew")
        txt.insert(tk.END, self._solutions_text())
        txt.config(state=tk.DISABLED)

    def _draw_schematic(self, canvas):
        """Draw a simple DC shunt motor schematic on a Canvas."""
        c = canvas
        c.delete("all")
        # Supply rails
        c.create_line(30, 60, 30, 340, width=2, fill="blue")    # +ve rail
        c.create_line(290, 60, 290, 340, width=2, fill="red")   # -ve rail
        c.create_text(20, 30, text="+", font=("Helvetica", 14, "bold"), fill="blue")
        c.create_text(290, 30, text="−", font=("Helvetica", 14, "bold"), fill="red")
        c.create_text(160, 15, text="Vt = 240 V", font=("Helvetica", 11, "bold"), fill="black")

        # Armature branch (series Ra + Ea)
        # Ra box
        c.create_rectangle(60, 110, 130, 150, outline="black", width=2)
        c.create_text(95, 130, text="Ra=4Ω", font=("Helvetica", 9))
        c.create_line(30, 130, 60, 130, width=2)

        # Ea (back-EMF circle)
        c.create_oval(155, 110, 225, 150, outline="black", width=2)
        c.create_text(190, 130, text="Ea", font=("Helvetica", 10, "italic"))
        c.create_line(130, 130, 155, 130, width=2)
        c.create_line(225, 130, 290, 130, width=2)

        # Armature (M circle)
        c.create_oval(130, 195, 220, 285, outline="black", width=3)
        c.create_text(175, 240, text="M", font=("Helvetica", 22, "bold"), fill="navy")
        c.create_text(175, 265, text="Armature", font=("Helvetica", 8))

        # Connect armature to rails
        c.create_line(30, 130, 30, 240, width=2)
        c.create_line(30, 240, 130, 240, width=2)
        c.create_line(220, 240, 290, 240, width=2)
        c.create_line(290, 130, 290, 240, width=2)

        # Field winding (shunt)
        c.create_rectangle(65, 310, 225, 345, outline="darkgreen", width=2)
        c.create_text(145, 327, text="Shunt Field Winding (Lf, Rf)", font=("Helvetica", 9), fill="darkgreen")
        c.create_line(30, 340, 65, 327, width=2, fill="darkgreen", dash=(4, 2))
        c.create_line(225, 327, 290, 340, width=2, fill="darkgreen", dash=(4, 2))

        # Labels
        c.create_text(145, 90, text="Armature Circuit", font=("Helvetica", 9, "italic"), fill="gray")
        c.create_text(145, 380, text="4-Pole Shunt-Wound DC Motor", font=("Helvetica", 10, "bold"))

    def _solutions_text(self):
        """Return formatted text with all problem solutions."""
        return """
════════════════════════════════════════════════════════════════════
  PROBLEM 28 – DC SHUNT MOTOR (Vt=240V, Ra=4Ω, n0=1800 r/min)
════════════════════════════════════════════════════════════════════

GIVEN:  4-pole shunt motor, Ra=4Ω, Vt=240V, n₀=1800 r/min
        At no-load: Ia≈0 → Vt = Ea → Ea = K_e × n₀

DERIVED CONSTANTS:
  K_e = Vt / n₀ = 240 / 1800 = 0.13333 V/(r/min)
  ω in rad/s  = n × (2π/60)
  K_e_rad = K_e / (2π/60) = 0.13333 × 9.5493 = 1.27324 V·s/rad
  K_t = K_e_rad = 1.27324 N·m/A   [SI, lossless coupling]

GENERAL CIRCUIT EQUATION:
  Vt = Ea + Ia×Ra
  Ia = (Vt - Ea) / Ra = (Vt - K_e_rad×ω) / Ra

────────────────────────────────────────────────────────────────────
Part (a) — Armature current at n = 900 r/min
────────────────────────────────────────────────────────────────────
  Ea = K_e × 900 = 0.13333 × 900 = 120.00 V
  Ia = (240 - 120) / 4 = 30.00 A   ✓

────────────────────────────────────────────────────────────────────
Part (b) — Mechanical power output at n = 1200 r/min
────────────────────────────────────────────────────────────────────
  Ea = 0.13333 × 1200 = 160.00 V
  Ia = (240 - 160) / 4 = 20.00 A
  P_mech = Ea × Ia = 160 × 20 = 3200 W
  P_mech = 3200 / 745.7 = 4.29 hp   ✓

────────────────────────────────────────────────────────────────────
Part (c) — Torque at n = 300 r/min
────────────────────────────────────────────────────────────────────
  Ea = 0.13333 × 300 = 40.00 V
  Ia = (240 - 40) / 4 = 50.00 A
  ω  = 300 × 2π/60 = 31.416 rad/s
  P_mech = 40 × 50 = 2000 W
  T = P_mech / ω = 2000 / 31.416 = 63.66 N·m
  Check: T = K_t × Ia = 1.27324 × 50 = 63.66 N·m   ✓

────────────────────────────────────────────────────────────────────
Part (d) — Starting torque (n = 0, Ea = 0)
────────────────────────────────────────────────────────────────────
  Ia_start = Vt / Ra = 240 / 4 = 60.00 A
  T_start  = K_t × Ia = 1.27324 × 60 = 76.39 N·m
  T_start  = 76.39 × 0.737562 = 56.35 ft·lbf   ✓

────────────────────────────────────────────────────────────────────
Part (e) — Torque-Speed Curve → see Tab 4
────────────────────────────────────────────────────────────────────
  Covers Quadrants 1, 2, and 4 (see Tab 4 for interactive plot)

════════════════════════════════════════════════════════════════════
  PROBLEM 29 – REDUCED VOLTAGE (60 V applied, same field)
════════════════════════════════════════════════════════════════════

GIVEN:  Same motor, same field current → K_e unchanged = 1.27324 V·s/rad
        Applied voltage Vt' = 60 V

Part (a) — No-load speed at 60 V:
  At no-load Ea = Vt' = 60 V
  n₀' = Vt' / K_e = 60 / 0.13333 = 450 r/min   ✓
  Torque-speed curve is parallel, shifted left (see Tab 4)

Part (b) — Frequency of armature current at n = 300 r/min:
  For a P-pole machine the electrical frequency of commutation:
    f = (P/2) × (n/60) = (4/2) × (300/60) = 2 × 5 = 10 Hz   ✓
  Formula: f = P × n / 120

════════════════════════════════════════════════════════════════════
  SUMMARY TABLE
════════════════════════════════════════════════════════════════════
  Speed    │ Ea (V) │ Ia (A) │ P_mech (W) │ Torque (N·m)
  ─────────┼────────┼────────┼────────────┼─────────────
  0 r/min  │   0.0  │  60.0  │      0     │  76.39 (start)
  300 rpm  │  40.0  │  50.0  │   2000     │  63.66
  600 rpm  │  80.0  │  40.0  │   3200     │  50.93
  900 rpm  │ 120.0  │  30.0  │   3600     │  38.20
 1200 rpm  │ 160.0  │  20.0  │   3200     │  25.46
 1500 rpm  │ 200.0  │  10.0  │   2000     │  12.73
 1800 rpm  │ 240.0  │   0.0  │      0     │   0.00 (no-load)
"""

    # =======================================================================
    # TAB 2 – Input Parameters and Controls
    # =======================================================================
    def _build_tab2(self):
        """Tab 2: Sliders for Vt, Ra, n0, poles with live readout."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 2 – Parameters ")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Motor Input Parameters",
                  font=("Helvetica", 13, "bold")).grid(row=0, column=0, columnspan=3,
                                                        pady=10, padx=8, sticky="w")

        def make_slider(row, label, var, lo, hi, fmt):
            ttk.Label(frame, text=label, width=28, anchor="e").grid(
                row=row, column=0, padx=8, pady=6, sticky="e")
            lbl_val = ttk.Label(frame, text=fmt.format(var.get()), width=12)
            lbl_val.grid(row=row, column=2, padx=8, sticky="w")

            def on_change(val, lv=lbl_val, f=fmt, v=var):
                lv.config(text=f.format(float(val)))

            sl = ttk.Scale(frame, from_=lo, to=hi, variable=var,
                           orient=tk.HORIZONTAL, command=on_change)
            sl.grid(row=row, column=1, sticky="ew", padx=8)
            return sl

        make_slider(1, "Supply voltage Vt [V]:",     self.var_vt,    1,   500, "{:.1f} V")
        make_slider(2, "Armature resistance Ra [Ω]:", self.var_ra,   0.1,  20, "{:.2f} Ω")
        make_slider(3, "No-load speed n₀ [r/min]:",  self.var_n0,  100, 5000, "{:.0f} rpm")

        # Poles combobox
        ttk.Label(frame, text="Number of poles:", width=28, anchor="e").grid(
            row=4, column=0, padx=8, pady=6, sticky="e")
        poles_cb = ttk.Combobox(frame, textvariable=self.var_poles,
                                values=[2, 4, 6, 8], width=6, state="readonly")
        poles_cb.grid(row=4, column=1, sticky="w", padx=8)
        poles_cb.current(1)  # default 4-pole

        # Reset button
        ttk.Button(frame, text="Reset to Defaults",
                   command=self._reset_params).grid(row=5, column=0, columnspan=3,
                                                    pady=14)

        # Live derived-constants display
        self._tab2_result = scrolledtext.ScrolledText(frame, height=14,
                                                      font=("Courier", 10))
        self._tab2_result.grid(row=6, column=0, columnspan=3, sticky="ew",
                               padx=8, pady=4)
        frame.rowconfigure(6, weight=1)

        # Update display on any parameter change
        for v in (self.var_vt, self.var_ra, self.var_n0):
            v.trace_add("write", lambda *_: self._update_tab2())
        self.var_poles.trace_add("write", lambda *_: self._update_tab2())
        self._update_tab2()

    def _reset_params(self):
        self.var_vt.set(DEFAULT_VT)
        self.var_ra.set(DEFAULT_RA)
        self.var_n0.set(DEFAULT_N0)
        self.var_poles.set(DEFAULT_POLES)

    def _update_tab2(self):
        vt, ra, n0, poles = self._params()
        Ke = vt / n0                              # V/(r/min)
        Ke_rad = ke_rad(vt, n0)                   # V·s/rad
        Ia_start = vt / ra
        T_start  = Ke_rad * Ia_start
        txt = (
            f"  Vt            = {vt:.2f} V\n"
            f"  Ra            = {ra:.3f} Ω\n"
            f"  n₀ (no-load)  = {n0:.1f} r/min\n"
            f"  Poles         = {poles}\n\n"
            f"  ── Derived Constants ──────────────────────────\n"
            f"  K_e           = {Ke:.6f} V/(r/min)\n"
            f"  K_e_rad (Kt)  = {Ke_rad:.6f} V·s/rad = N·m/A\n"
            f"  ω₀ (no-load)  = {n0*2*math.pi/60:.4f} rad/s\n\n"
            f"  ── Starting Conditions (n=0) ──────────────────\n"
            f"  Ia_start      = {Ia_start:.3f} A\n"
            f"  T_start       = {T_start:.3f} N·m = {T_start*FTLBF_PER_NM:.3f} ft·lbf\n\n"
            f"  ── Problem 29 (Vt' = 60 V, same field) ────────\n"
            f"  n₀' (60V)     = {60/Ke:.2f} r/min  [Ea=60V at no-load]\n"
            f"  f at 300 rpm  = {poles*300/120:.2f} Hz  [f=P×n/120]\n"
        )
        self._tab2_result.config(state=tk.NORMAL)
        self._tab2_result.delete("1.0", tk.END)
        self._tab2_result.insert(tk.END, txt)
        self._tab2_result.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 3 – Differential Equations and Dynamic Simulation
    # =======================================================================
    def _build_tab3(self):
        """Tab 3: solve_ivp dynamic simulation with Ia(t), ω(t), T(t)."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 3 – Simulation ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Dynamic Simulation – Armature + Mechanical ODEs",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")

        # ---- Plot area -------------------------------------------------------
        plot_frame = ttk.LabelFrame(frame, text="Simulation Plots")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig3 = Figure(figsize=(7, 6))
        self._ax3_ia, self._ax3_om, self._ax3_t = \
            self._fig3.subplots(3, 1, sharex=True)
        self._fig3.tight_layout(pad=2.5)
        self._canvas3, _ = embed_figure(self._fig3, plot_frame)

        # ---- Control panel ---------------------------------------------------
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)

        ttk.Label(ctrl, text="ODE Parameters", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_La    = tk.DoubleVar(value=0.05)   # Armature inductance [H]
        self._v_J     = tk.DoubleVar(value=0.02)   # Moment of inertia [kg·m²]
        self._v_B     = tk.DoubleVar(value=0.005)  # Friction [N·m·s/rad]
        self._v_Tload = tk.DoubleVar(value=5.0)    # Load torque [N·m]
        self._v_tsim  = tk.DoubleVar(value=2.0)    # Simulation time [s]

        sliders3 = [
            ("La [H]:",           self._v_La,    0.001, 0.5,   0.001, "{:.3f} H"),
            ("J [kg·m²]:",        self._v_J,     0.001, 1.0,   0.001, "{:.3f} kg·m²"),
            ("B [N·m·s/rad]:",    self._v_B,     0.0,   0.1,   0.001, "{:.4f}"),
            ("T_load [N·m]:",     self._v_Tload, 0.0,  80.0,   0.5,   "{:.1f} N·m"),
            ("Sim time [s]:",     self._v_tsim,  0.1,  10.0,   0.1,   "{:.1f} s"),
        ]
        for lbl, var, lo, hi, res, fmt in sliders3:
            ttk.Label(ctrl, text=lbl, anchor="w").pack(fill=tk.X, padx=4)
            lbl_v = ttk.Label(ctrl, text=fmt.format(var.get()))
            lbl_v.pack()
            def _cb(val, lv=lbl_v, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=_cb).pack(fill=tk.X, padx=6, pady=2)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(pady=8)
        ttk.Button(btn_frame, text="▶ Start", command=self._sim_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="■ Stop",  command=self._sim_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="↺ Reset", command=self._sim_reset).pack(side=tk.LEFT, padx=4)

        self._tab3_status = ttk.Label(ctrl, text="Ready.", foreground="green")
        self._tab3_status.pack(pady=4)

        self._result3 = scrolledtext.ScrolledText(ctrl, height=8, font=("Courier", 9))
        self._result3.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _ode_system(self, t, y, vt, ra, La, Ke_rad, J, B, Tload):
        """
        State: y = [Ia, omega]
        dIa/dt    = (Vt - Ra*Ia - Ke_rad*omega) / La
        domega/dt = (Ke_rad*Ia - B*omega - Tload) / J
        """
        Ia, omega = y
        omega = max(omega, -1e4)
        dIa_dt    = (vt - ra * Ia - Ke_rad * omega) / La
        domega_dt = (Ke_rad * Ia - B * omega - Tload) / J
        return [dIa_dt, domega_dt]

    def _sim_start(self):
        if self._sim_running:
            return
        self._sim_running = True
        self._tab3_status.config(text="Simulating…", foreground="blue")
        self.update_idletasks()
        try:
            vt, ra, n0, _ = self._params()
            La    = clamp(float(self._v_La.get()),    1e-4, 10.0)
            J     = clamp(float(self._v_J.get()),     1e-4, 100.0)
            B     = clamp(float(self._v_B.get()),     0.0,  10.0)
            Tload = clamp(float(self._v_Tload.get()), 0.0,  500.0)
            tsim  = clamp(float(self._v_tsim.get()),  0.01, 60.0)
            Ke    = ke_rad(vt, n0)

            sol = solve_ivp(
                self._ode_system,
                [0, tsim],
                [0.0, 0.0],
                args=(vt, ra, La, Ke, J, B, Tload),
                method="RK45",
                max_step=tsim / 1000,
                dense_output=True,
            )
            t   = sol.t
            Ia  = sol.y[0]
            om  = sol.y[1]
            T   = Ke * Ia
            n_rpm = om * 60 / (2 * math.pi)

            # Plot
            for ax in (self._ax3_ia, self._ax3_om, self._ax3_t):
                ax.cla()
            self._ax3_ia.plot(t, Ia, color="C0")
            self._ax3_ia.set_ylabel("Ia (A)")
            self._ax3_ia.grid(True)
            self._ax3_om.plot(t, n_rpm, color="C1")
            self._ax3_om.set_ylabel("Speed (rpm)")
            self._ax3_om.grid(True)
            self._ax3_t.plot(t, T, color="C2")
            self._ax3_t.set_ylabel("Torque (N·m)")
            self._ax3_t.set_xlabel("Time (s)")
            self._ax3_t.grid(True)
            self._fig3.tight_layout(pad=2.5)
            self._canvas3.draw()

            # Results
            ss_idx = -1
            info = (
                f"Steady-state (t={t[ss_idx]:.2f}s):\n"
                f"  Ia_ss    = {Ia[ss_idx]:.3f} A\n"
                f"  speed_ss = {n_rpm[ss_idx]:.1f} rpm\n"
                f"  T_ss     = {T[ss_idx]:.3f} N·m\n"
                f"Peak Ia    = {np.max(Ia):.3f} A\n"
                f"Peak T     = {np.max(T):.3f} N·m\n"
            )
            self._result3.config(state=tk.NORMAL)
            self._result3.delete("1.0", tk.END)
            self._result3.insert(tk.END, info)
            self._result3.config(state=tk.DISABLED)
            self._tab3_status.config(text="Done.", foreground="green")
        except Exception as e:
            self._tab3_status.config(text=f"Error: {e}", foreground="red")
        finally:
            self._sim_running = False

    def _sim_stop(self):
        self._sim_running = False
        self._tab3_status.config(text="Stopped.", foreground="orange")

    def _sim_reset(self):
        self._sim_running = False
        for ax in (self._ax3_ia, self._ax3_om, self._ax3_t):
            ax.cla()
        self._canvas3.draw()
        self._tab3_status.config(text="Reset.", foreground="gray")

    # =======================================================================
    # TAB 4 – Torque-Speed Curves (28e and 29a)
    # =======================================================================
    def _build_tab4(self):
        """Tab 4: Torque-speed curves for 240V and 60V, all quadrants."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 4 – T-n Curves ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Torque-Speed Characteristics – Quadrants 1, 2, 4",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="T-n Plot")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig4 = Figure(figsize=(7, 5))
        self._ax4 = self._fig4.add_subplot(111)
        self._canvas4, _ = embed_figure(self._fig4, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="Voltage Sweep", font=("Helvetica", 10, "bold")).pack(pady=4)
        self._v_sweep = tk.DoubleVar(value=120.0)
        lbl_sweep = ttk.Label(ctrl, text="120.0 V")
        lbl_sweep.pack()
        def _cb_sweep(val, lv=lbl_sweep):
            lv.config(text=f"{float(val):.1f} V")
        ttk.Scale(ctrl, from_=10, to=500, variable=self._v_sweep,
                  orient=tk.HORIZONTAL, command=_cb_sweep).pack(fill=tk.X, padx=6)
        ttk.Button(ctrl, text="Update Plot", command=self._update_tab4).pack(pady=8)
        ttk.Button(ctrl, text="Show 240V & 60V", command=self._plot_240_60).pack(pady=4)

        self._result4 = scrolledtext.ScrolledText(ctrl, height=12, font=("Courier", 9))
        self._result4.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._plot_240_60()  # Initial plot

    def _update_tab4(self):
        vt, ra, n0, _ = self._params()
        v_sweep = clamp(float(self._v_sweep.get()), 1.0, 2000.0)
        n_arr = np.linspace(-n0 * 0.3, n0 * 1.1, 500)
        _, T_main  = torque_speed_line(vt, ra, n0, n_arr)
        _, T_sweep = torque_speed_line(v_sweep, ra, n0, n_arr)
        ax = self._ax4
        ax.cla()
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.plot(n_arr, T_main,  label=f"{vt:.0f} V (nominal)", color="C0")
        ax.plot(n_arr, T_sweep, label=f"{v_sweep:.0f} V (sweep)", color="C3",
                linestyle="--")
        # Operating points from Problem 28
        pts = [(900, 38.20), (1200, 25.46), (300, 63.66), (0, 76.39)]
        for (n_pt, t_pt) in pts:
            ax.annotate(f"{n_pt}rpm\n{t_pt:.1f}N·m",
                        xy=(n_pt, t_pt), xytext=(n_pt+60, t_pt+3),
                        fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8),
                        color="C0")
            ax.plot(n_pt, t_pt, "o", color="C0", markersize=5)
        ax.set_xlabel("Speed (r/min)")
        ax.set_ylabel("Torque (N·m)")
        ax.set_title("DC Shunt Motor – Torque-Speed Curves")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.5)
        self._fig4.tight_layout()
        self._canvas4.draw()

    def _plot_240_60(self):
        vt, ra, n0, _ = self._params()
        n_arr = np.linspace(-n0 * 0.3, n0 * 1.1, 600)
        _, T240 = torque_speed_line(240.0, ra, n0, n_arr)
        _, T60  = torque_speed_line(60.0,  ra, n0, n_arr)
        ax = self._ax4
        ax.cla()
        ax.axhline(0, color="k", linewidth=0.8)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.plot(n_arr, T240, label="240 V (nominal)", color="C0", linewidth=2)
        ax.plot(n_arr, T60,  label="60 V (Prob 29)",  color="C1", linewidth=2,
                linestyle="--")

        # Annotate quadrants
        ax.text(n0*0.5,  max(T240)*0.7,  "Q1\n(Motor)", fontsize=9, color="green", ha="center")
        ax.text(-n0*0.15, max(T240)*0.7, "Q2\n(Braking)", fontsize=9, color="orange", ha="center")
        ax.text(n0*0.5,  min(T240)*0.7,  "Q4\n(Gen)", fontsize=9, color="red", ha="center")

        # Problem 28 operating points
        pts28 = {"28a (900rpm)": (900, 38.20), "28b (1200rpm)": (1200, 25.46),
                 "28c (300rpm)": (300, 63.66)}
        colors = ["C2", "C3", "C4"]
        for (lbl, (nn, tt)), col in zip(pts28.items(), colors):
            ax.plot(nn, tt, "s", color=col, markersize=7, label=lbl)

        ax.set_xlabel("Speed (r/min)")
        ax.set_ylabel("Torque (N·m)")
        ax.set_title("Torque-Speed: 240V vs 60V (Problems 28 & 29)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.5)
        self._fig4.tight_layout()
        self._canvas4.draw()

        # Text summary
        Ke = vt / n0
        info = (
            "Problem 28 – 240V results:\n"
            "  Speed  Ea(V)  Ia(A)  T(N·m)\n"
        )
        for nn in [0, 300, 600, 900, 1200, 1500, 1800]:
            Ea = Ke * nn
            Ia = (240 - Ea) / ra
            T  = ke_rad(240, n0) * Ia
            info += f"  {nn:5d}  {Ea:5.1f}  {Ia:5.1f}  {T:6.2f}\n"
        info += "\nProblem 29 – 60V results:\n"
        info += f"  No-load speed = {60/Ke:.1f} rpm\n"
        info += "  f at 300rpm = 10.0 Hz\n"
        self._result4.config(state=tk.NORMAL)
        self._result4.delete("1.0", tk.END)
        self._result4.insert(tk.END, info)
        self._result4.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 5 – Fault Current Analysis
    # =======================================================================
    def _build_tab5(self):
        """Tab 5: Short-circuit fault current transient analysis."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 5 – Fault Current ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Fault Current Analysis – Short-Circuit Transient",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="Ia(t) Fault Response")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig5 = Figure(figsize=(7, 4))
        self._ax5 = self._fig5.add_subplot(111)
        self._canvas5, _ = embed_figure(self._fig5, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="Fault Parameters", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_La_fault = tk.DoubleVar(value=0.05)
        self._v_t_fault  = tk.DoubleVar(value=0.5)

        def _make_sl5(lbl, var, lo, hi, fmt):
            ttk.Label(ctrl, text=lbl).pack()
            lv = ttk.Label(ctrl, text=fmt.format(var.get()))
            lv.pack()
            def cb(val, lv=lv, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=cb).pack(fill=tk.X, padx=6, pady=2)

        _make_sl5("La [H]:",    self._v_La_fault, 0.001, 1.0, "{:.3f} H")
        _make_sl5("Duration [s]:", self._v_t_fault, 0.01, 2.0, "{:.2f} s")

        ttk.Button(ctrl, text="▶ Run Fault", command=self._run_fault).pack(pady=8)

        self._result5 = scrolledtext.ScrolledText(ctrl, height=14, font=("Courier", 9))
        self._result5.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _run_fault(self):
        try:
            vt, ra, n0, _ = self._params()
            La  = clamp(float(self._v_La_fault.get()), 1e-5, 100.0)
            dur = clamp(float(self._v_t_fault.get()),  1e-3, 60.0)
            # Ia(t) = (Vt/Ra)(1 - exp(-Ra*t/La))  [motor stalled at Ea=0]
            t = np.linspace(0, dur, 1000)
            tau = La / ra
            Ia_inf = vt / ra
            Ia = Ia_inf * (1.0 - np.exp(-t / tau))

            # I²t energy
            I2t = np.trapz(Ia**2, t)

            ax = self._ax5
            ax.cla()
            ax.plot(t, Ia, color="C3", linewidth=2, label="Fault Ia(t)")
            ax.axhline(Ia_inf, color="gray", linestyle="--", label=f"Steady {Ia_inf:.1f} A")
            ax.axhline(Ia_inf * 0.632, color="C0", linestyle=":",
                       label=f"τ ({tau*1000:.1f} ms) → {Ia_inf*0.632:.1f} A")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Ia (A)")
            ax.set_title("Short-Circuit Fault Current (Ea=0, stalled)")
            ax.legend(fontsize=8)
            ax.grid(True)
            self._fig5.tight_layout()
            self._canvas5.draw()

            info = (
                f"Supply voltage   Vt  = {vt:.1f} V\n"
                f"Armature resist. Ra  = {ra:.3f} Ω\n"
                f"Armature induct. La  = {La*1000:.1f} mH\n"
                f"Time constant    τ   = {tau*1000:.2f} ms\n"
                f"Peak/steady-state Ia = {Ia_inf:.2f} A\n"
                f"I²t energy          = {I2t:.2f} A²·s\n"
                f"Duration            = {dur:.2f} s\n"
                f"\nNote: Peak current at t=0⁺ depends on\n"
                f"stray inductance. Ia(∞)=Vt/Ra is the\n"
                f"limiting thermal-fault condition.\n"
            )
            self._result5.config(state=tk.NORMAL)
            self._result5.delete("1.0", tk.END)
            self._result5.insert(tk.END, info)
            self._result5.config(state=tk.DISABLED)
        except Exception as e:
            self._result5.config(state=tk.NORMAL)
            self._result5.insert(tk.END, f"\nError: {e}\n")
            self._result5.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 6 – Protection Coordination
    # =======================================================================
    def _build_tab6(self):
        """Tab 6: Overcurrent relay, fuse, and thermal-damage curves."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 6 – Protection ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Protection Coordination – Relay / Fuse / Thermal Damage",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="Log-Log Protection Curves")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig6 = Figure(figsize=(7, 5))
        self._ax6 = self._fig6.add_subplot(111)
        self._canvas6, _ = embed_figure(self._fig6, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="Relay Settings", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_pickup  = tk.DoubleVar(value=30.0)   # Pickup current [A]
        self._v_td      = tk.DoubleVar(value=1.0)    # Time dial

        def _sl6(lbl, var, lo, hi, fmt):
            ttk.Label(ctrl, text=lbl).pack()
            lv = ttk.Label(ctrl, text=fmt.format(var.get()))
            lv.pack()
            def cb(val, lv=lv, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=cb).pack(fill=tk.X, padx=6, pady=2)

        _sl6("Pickup current [A]:", self._v_pickup, 5, 120, "{:.1f} A")
        _sl6("Time dial (TD):",     self._v_td,     0.1, 10,  "{:.2f}")

        ttk.Button(ctrl, text="Update", command=self._update_tab6).pack(pady=8)
        self._result6 = scrolledtext.ScrolledText(ctrl, height=14, font=("Courier", 9))
        self._result6.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._update_tab6()

    def _update_tab6(self):
        try:
            vt, ra, _, _ = self._params()
            Ip = clamp(float(self._v_pickup.get()), 1.0, 1e4)
            TD = clamp(float(self._v_td.get()), 0.01, 100.0)

            I_sc = vt / ra  # bolted fault current
            I_range = np.logspace(np.log10(max(Ip * 0.8, 1.0)),
                                  np.log10(I_sc * 3), 500)
            M = I_range / Ip   # multiples of pickup

            # IEC Standard Inverse relay curve: t = TD * 0.14 / (M^0.02 - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_relay = TD * 0.14 / (np.maximum(M, 1.001)**0.02 - 1)

            # Fuse curve: approximate T-link  t = k / I^2
            k_fuse = 50 * Ip**2   # normalisation constant
            t_fuse = k_fuse / I_range**2

            # Thermal damage curve: t = K / I^2  (motor thermal limit)
            K_therm = 200.0 * (Ip)**2
            t_therm = K_therm / I_range**2

            ax = self._ax6
            ax.cla()
            ax.loglog(I_range, t_relay, label="IEC Relay (std inverse)", color="C0", linewidth=2)
            ax.loglog(I_range, t_fuse,  label="Fuse (T-link approx)",    color="C3", linewidth=2,
                      linestyle="--")
            ax.loglog(I_range, t_therm, label="Motor thermal limit",     color="C2", linewidth=2,
                      linestyle=":")
            ax.axvline(Ip,   color="gray", linestyle="-.", linewidth=1, label=f"Pickup {Ip:.0f}A")
            ax.axvline(I_sc, color="red",  linestyle="-.", linewidth=1, label=f"Isc {I_sc:.0f}A")
            ax.set_xlabel("Current (A)")
            ax.set_ylabel("Trip time (s)")
            ax.set_title("Protection Coordination (Log-Log)")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.4)
            ax.set_xlim(I_range[0], I_range[-1])
            ax.set_ylim(1e-3, 1e3)
            self._fig6.tight_layout()
            self._canvas6.draw()

            # Check coordination at Isc
            idx = np.argmin(np.abs(I_range - I_sc))
            t_r_sc = t_relay[idx]
            t_f_sc = t_fuse[idx]
            coordinated = "PASS ✓" if t_r_sc < t_f_sc else "FAIL ✗"
            info = (
                f"Pickup current  = {Ip:.1f} A\n"
                f"Time dial       = {TD:.2f}\n"
                f"Fault current   = {I_sc:.1f} A\n"
                f"Relay @ Isc     = {t_r_sc:.4f} s\n"
                f"Fuse @ Isc      = {t_f_sc:.4f} s\n"
                f"Coordination    = {coordinated}\n"
                f"\n(Relay must trip BEFORE fuse melts\n"
                f" to allow resetting)\n"
            )
            self._result6.config(state=tk.NORMAL)
            self._result6.delete("1.0", tk.END)
            self._result6.insert(tk.END, info)
            self._result6.config(state=tk.DISABLED)
        except Exception as e:
            self._result6.config(state=tk.NORMAL)
            self._result6.insert(tk.END, f"\nError: {e}\n")
            self._result6.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 7 – Speed Controller (PID + Fuzzy)
    # =======================================================================
    def _build_tab7(self):
        """Tab 7: PID and Fuzzy speed controllers with step-load response."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 7 – Controller ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Speed Controller – PID vs Fuzzy Logic",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="Speed Response")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig7 = Figure(figsize=(7, 5))
        self._ax7a, self._ax7b = self._fig7.subplots(2, 1, sharex=True)
        self._fig7.tight_layout(pad=2.5)
        self._canvas7, _ = embed_figure(self._fig7, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="PID Gains", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_Kp = tk.DoubleVar(value=2.0)
        self._v_Ki = tk.DoubleVar(value=0.5)
        self._v_Kd = tk.DoubleVar(value=0.1)
        self._v_sp = tk.DoubleVar(value=1200.0)
        self._v_Tl7 = tk.DoubleVar(value=10.0)
        self._v_ts7 = tk.DoubleVar(value=3.0)

        sliders7 = [
            ("Kp:",             self._v_Kp,  0.0, 20.0, "{:.2f}"),
            ("Ki:",             self._v_Ki,  0.0,  5.0, "{:.3f}"),
            ("Kd:",             self._v_Kd,  0.0,  2.0, "{:.3f}"),
            ("Setpoint [rpm]:", self._v_sp,  10, 3000, "{:.0f} rpm"),
            ("Step load [N·m]:",self._v_Tl7, 0, 80,  "{:.1f} N·m"),
            ("Sim time [s]:",   self._v_ts7, 0.5, 15, "{:.1f} s"),
        ]
        for lbl, var, lo, hi, fmt in sliders7:
            ttk.Label(ctrl, text=lbl, anchor="w").pack(fill=tk.X, padx=4)
            lv = ttk.Label(ctrl, text=fmt.format(var.get()))
            lv.pack()
            def cb(val, lv=lv, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=cb).pack(fill=tk.X, padx=6, pady=1)

        btn7 = ttk.Frame(ctrl)
        btn7.pack(pady=6)
        ttk.Button(btn7, text="▶ Run", command=self._run_pid).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn7, text="↺ Reset", command=self._reset_pid).pack(side=tk.LEFT, padx=3)

        self._result7 = scrolledtext.ScrolledText(ctrl, height=8, font=("Courier", 9))
        self._result7.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _pid_sim(self, vt, ra, n0, Kp, Ki, Kd, setpoint_rpm, Tload, tsim):
        """Discrete-time PID simulation on motor ODE."""
        La  = 0.05
        J   = 0.02
        B   = 0.005
        Ke  = ke_rad(vt, n0)
        dt  = 1e-3
        N   = int(tsim / dt)
        t   = np.linspace(0, tsim, N)
        omega_sp = setpoint_rpm * 2 * math.pi / 60

        Ia = 0.0; omega = 0.0
        integral = 0.0; prev_err = 0.0
        omega_arr = np.zeros(N)
        Ia_arr    = np.zeros(N)
        vt_cmd    = np.zeros(N)

        for i in range(N):
            # Step load at half time
            Tl = Tload if t[i] > tsim / 2 else 0.0
            err = omega_sp - omega
            integral   = clamp(integral + err * dt, -200, 200)   # anti-windup
            derivative = (err - prev_err) / dt
            vt_eff = clamp(Kp * err + Ki * integral + Kd * derivative, 0, vt * 2)
            prev_err = err
            dIa    = (vt_eff - ra * Ia - Ke * omega) / La
            domega = (Ke * Ia - B * omega - Tl) / J
            Ia    = clamp(Ia + dIa * dt, -1e4, 1e4)
            omega = clamp(omega + domega * dt, -1e5, 1e5)
            omega_arr[i] = omega * 60 / (2 * math.pi)
            Ia_arr[i]    = Ia
            vt_cmd[i]    = vt_eff

        return t, omega_arr, Ia_arr, vt_cmd

    def _fuzzy_control(self, error, d_error, vt):
        """Simple 5-rule Mamdani-style fuzzy controller."""
        # Normalise
        e_n = clamp(error / 200.0, -1, 1)
        de_n = clamp(d_error / 500.0, -1, 1)
        # Output voltage correction (normalised)
        u = clamp(2.5 * e_n + 0.5 * de_n, -1, 1)
        return u * vt

    def _fuzzy_sim(self, vt, ra, n0, setpoint_rpm, Tload, tsim):
        La = 0.05; J = 0.02; B = 0.005
        Ke = ke_rad(vt, n0)
        dt = 1e-3; N = int(tsim / dt)
        t  = np.linspace(0, tsim, N)
        omega_sp = setpoint_rpm * 2 * math.pi / 60
        Ia = 0.0; omega = 0.0; prev_err = 0.0
        omega_arr = np.zeros(N)

        for i in range(N):
            Tl  = Tload if t[i] > tsim / 2 else 0.0
            err = omega_sp - omega
            d_err = (err - prev_err) / dt
            vt_eff = clamp(self._fuzzy_control(err, d_err, vt), 0, vt * 2)
            prev_err = err
            dIa    = (vt_eff - ra * Ia - Ke * omega) / La
            domega = (Ke * Ia - B * omega - Tl) / J
            Ia     = clamp(Ia + dIa * dt, -1e4, 1e4)
            omega  = clamp(omega + domega * dt, -1e5, 1e5)
            omega_arr[i] = omega * 60 / (2 * math.pi)

        return t, omega_arr

    def _run_pid(self):
        try:
            vt, ra, n0, _ = self._params()
            Kp  = float(self._v_Kp.get())
            Ki  = float(self._v_Ki.get())
            Kd  = float(self._v_Kd.get())
            sp  = clamp(float(self._v_sp.get()), 1, n0 * 2)
            Tl  = clamp(float(self._v_Tl7.get()), 0, 500)
            ts  = clamp(float(self._v_ts7.get()), 0.1, 60)

            t, n_pid, Ia_pid, _ = self._pid_sim(vt, ra, n0, Kp, Ki, Kd, sp, Tl, ts)
            t2, n_fuz = self._fuzzy_sim(vt, ra, n0, sp, Tl, ts)

            for ax in (self._ax7a, self._ax7b):
                ax.cla()
            self._ax7a.plot(t, n_pid, color="C0", label="PID")
            self._ax7a.plot(t2, n_fuz, color="C3", linestyle="--", label="Fuzzy")
            self._ax7a.axhline(sp, color="k", linestyle=":", linewidth=0.8, label=f"SP {sp:.0f}rpm")
            self._ax7a.axvline(ts/2, color="gray", linestyle="-.", linewidth=0.8)
            self._ax7a.set_ylabel("Speed (rpm)")
            self._ax7a.legend(fontsize=8)
            self._ax7a.grid(True)
            self._ax7b.plot(t, Ia_pid, color="C0")
            self._ax7b.set_ylabel("Ia_PID (A)")
            self._ax7b.set_xlabel("Time (s)")
            self._ax7b.grid(True)
            self._fig7.tight_layout(pad=2.5)
            self._canvas7.draw()

            # Metrics
            half = len(t) // 2
            pid_os  = (max(n_pid) - sp) / sp * 100
            pid_ss  = abs(n_pid[-1] - sp)
            fuz_os  = (max(n_fuz) - sp) / sp * 100
            fuz_ss  = abs(n_fuz[-1] - sp)
            info = (
                f"Setpoint = {sp:.0f} rpm\n"
                f"Load step at t={ts/2:.1f}s → {Tl:.1f} N·m\n\n"
                f"── PID Controller ──────────────\n"
                f"  Overshoot   = {pid_os:.2f}%\n"
                f"  SS error    = {pid_ss:.2f} rpm\n\n"
                f"── Fuzzy Controller ────────────\n"
                f"  Overshoot   = {fuz_os:.2f}%\n"
                f"  SS error    = {fuz_ss:.2f} rpm\n"
            )
            self._result7.config(state=tk.NORMAL)
            self._result7.delete("1.0", tk.END)
            self._result7.insert(tk.END, info)
            self._result7.config(state=tk.DISABLED)
        except Exception as e:
            self._result7.config(state=tk.NORMAL)
            self._result7.insert(tk.END, f"\nError: {e}\n")
            self._result7.config(state=tk.DISABLED)

    def _reset_pid(self):
        for ax in (self._ax7a, self._ax7b):
            ax.cla()
        self._canvas7.draw()

    # =======================================================================
    # TAB 8 – Thermal Analysis
    # =======================================================================
    def _build_tab8(self):
        """Tab 8: Lumped thermal model, temperature rise, duty cycles."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 8 – Thermal ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Thermal Analysis – Lumped RC Thermal Model",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="Temperature Rise θ(t)")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig8 = Figure(figsize=(7, 4))
        self._ax8 = self._fig8.add_subplot(111)
        self._canvas8, _ = embed_figure(self._fig8, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="Thermal Parameters", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_Rth   = tk.DoubleVar(value=1.5)    # Thermal resistance [°C/W]
        self._v_Cth   = tk.DoubleVar(value=200.0)  # Thermal capacitance [J/°C]
        self._v_theta_amb = tk.DoubleVar(value=25.0)
        self._v_theta_max = tk.DoubleVar(value=120.0)
        self._v_Ia8   = tk.DoubleVar(value=30.0)
        self._v_duty  = tk.DoubleVar(value=0.8)
        self._v_ts8   = tk.DoubleVar(value=300.0)

        def _sl8(lbl, var, lo, hi, fmt):
            ttk.Label(ctrl, text=lbl).pack()
            lv = ttk.Label(ctrl, text=fmt.format(var.get()))
            lv.pack()
            def cb(val, lv=lv, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=cb).pack(fill=tk.X, padx=6, pady=1)

        _sl8("Rth [°C/W]:",     self._v_Rth,       0.1, 10, "{:.2f}")
        _sl8("Cth [J/°C]:",     self._v_Cth,       10, 2000, "{:.0f}")
        _sl8("Tamb [°C]:",      self._v_theta_amb,  0, 60,  "{:.0f} °C")
        _sl8("Tmax [°C]:",      self._v_theta_max,  50, 200, "{:.0f} °C")
        _sl8("Ia [A]:",         self._v_Ia8,         0, 100, "{:.1f} A")
        _sl8("Duty cycle:",     self._v_duty,       0.1, 1.0, "{:.2f}")
        _sl8("Duration [s]:",   self._v_ts8,         30, 3600, "{:.0f} s")

        ttk.Button(ctrl, text="▶ Run", command=self._run_thermal).pack(pady=8)
        self._result8 = scrolledtext.ScrolledText(ctrl, height=8, font=("Courier", 9))
        self._result8.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    def _run_thermal(self):
        try:
            _, ra, _, _ = self._params()
            Rth   = clamp(float(self._v_Rth.get()), 0.01, 1e4)
            Cth   = clamp(float(self._v_Cth.get()), 1.0,  1e5)
            Tamb  = clamp(float(self._v_theta_amb.get()), -50, 200)
            Tmax  = clamp(float(self._v_theta_max.get()), 50, 500)
            Ia    = clamp(float(self._v_Ia8.get()), 0, 1e4)
            duty  = clamp(float(self._v_duty.get()), 0.01, 1.0)
            tsim  = clamp(float(self._v_ts8.get()), 1, 86400)
            tau   = Rth * Cth
            P_avg = duty * (Ia**2 * ra)   # average copper loss [W]
            # θ(t) = Tamb + Rth*P_avg*(1 - exp(-t/τ))
            t = np.linspace(0, tsim, 1000)
            theta = Tamb + Rth * P_avg * (1 - np.exp(-t / tau))
            theta_ss = Tamb + Rth * P_avg

            ax = self._ax8
            ax.cla()
            ax.plot(t, theta, color="C3", linewidth=2, label="θ(t)")
            ax.axhline(theta_ss, color="gray", linestyle="--",
                       label=f"SS: {theta_ss:.1f}°C")
            ax.axhline(Tmax, color="red", linestyle=":", linewidth=2,
                       label=f"Tmax {Tmax:.0f}°C")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Motor Temperature Rise")
            ax.legend(fontsize=8)
            ax.grid(True)
            self._fig8.tight_layout()
            self._canvas8.draw()

            exceed = "YES – OVERHEATING ⚠" if theta_ss > Tmax else "NO – Within limit ✓"
            info = (
                f"P_loss (avg)  = {P_avg:.2f} W\n"
                f"τ (thermal)   = {tau:.1f} s\n"
                f"θ_ss          = {theta_ss:.1f} °C\n"
                f"T_max         = {Tmax:.0f} °C\n"
                f"Exceeds limit = {exceed}\n"
                f"Duty cycle    = {duty*100:.0f}%\n"
            )
            self._result8.config(state=tk.NORMAL)
            self._result8.delete("1.0", tk.END)
            self._result8.insert(tk.END, info)
            self._result8.config(state=tk.DISABLED)
        except Exception as e:
            self._result8.config(state=tk.NORMAL)
            self._result8.insert(tk.END, f"\nError: {e}\n")
            self._result8.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 9 – Economic Analysis
    # =======================================================================
    def _build_tab9(self):
        """Tab 9: Efficiency vs speed, loss breakdown, annual cost."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 9 – Economics ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Economic Analysis – Efficiency & Energy Cost",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="Efficiency & Loss Curves")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig9 = Figure(figsize=(7, 5))
        self._ax9a, self._ax9b = self._fig9.subplots(2, 1, sharex=True)
        self._fig9.tight_layout(pad=2.5)
        self._canvas9, _ = embed_figure(self._fig9, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="Cost Parameters", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_cost_kwh   = tk.DoubleVar(value=0.15)   # $/kWh
        self._v_hours_yr   = tk.DoubleVar(value=4000.0) # operating h/year
        self._v_Pcore      = tk.DoubleVar(value=50.0)   # core loss [W]
        self._v_Pmech      = tk.DoubleVar(value=30.0)   # mechanical loss [W]

        def _sl9(lbl, var, lo, hi, fmt):
            ttk.Label(ctrl, text=lbl).pack()
            lv = ttk.Label(ctrl, text=fmt.format(var.get()))
            lv.pack()
            def cb(val, lv=lv, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=cb).pack(fill=tk.X, padx=6, pady=1)

        _sl9("Cost [$/kWh]:",    self._v_cost_kwh, 0.01, 1.0, "${:.3f}")
        _sl9("Hours/year:",      self._v_hours_yr, 100, 8760, "{:.0f} h")
        _sl9("Core loss [W]:",   self._v_Pcore,    0, 500, "{:.0f} W")
        _sl9("Mech loss [W]:",   self._v_Pmech,    0, 500, "{:.0f} W")

        ttk.Button(ctrl, text="Update", command=self._update_tab9).pack(pady=8)
        self._result9 = scrolledtext.ScrolledText(ctrl, height=12, font=("Courier", 9))
        self._result9.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._update_tab9()

    def _update_tab9(self):
        try:
            vt, ra, n0, _ = self._params()
            cost   = clamp(float(self._v_cost_kwh.get()), 1e-4, 100)
            hrs    = clamp(float(self._v_hours_yr.get()), 1, 8760)
            Pcore  = clamp(float(self._v_Pcore.get()), 0, 1e4)
            Pmech  = clamp(float(self._v_Pmech.get()), 0, 1e4)
            Ke     = ke_rad(vt, n0)

            n_arr = np.linspace(10, n0 * 1.05, 400)
            omega = n_arr * 2 * math.pi / 60
            Ea    = Ke * omega
            Ia    = (vt - Ea) / ra
            P_in  = vt * np.maximum(Ia, 0)
            P_cu  = Ia**2 * ra
            P_out = np.maximum(Ea * Ia, 0)
            P_tot_loss = P_cu + Pcore + Pmech
            eff   = np.where(P_in > 0, P_out / P_in * 100, 0.0)
            eff   = np.clip(eff, 0, 100)

            ax1, ax2 = self._ax9a, self._ax9b
            ax1.cla(); ax2.cla()
            ax1.plot(n_arr, eff, color="C2", linewidth=2, label="Efficiency %")
            ax1.set_ylabel("Efficiency (%)")
            ax1.legend(fontsize=8)
            ax1.grid(True)
            ax2.stackplot(n_arr, P_cu,
                          np.full_like(n_arr, Pcore),
                          np.full_like(n_arr, Pmech),
                          labels=["Copper", "Core", "Mech"],
                          colors=["C3", "C1", "C0"])
            ax2.set_ylabel("Loss (W)")
            ax2.set_xlabel("Speed (rpm)")
            ax2.legend(fontsize=8, loc="upper left")
            ax2.grid(True)
            self._fig9.tight_layout(pad=2.5)
            self._canvas9.draw()

            # Find peak efficiency
            pk_idx = np.argmax(eff)
            pk_n   = n_arr[pk_idx]
            pk_eff = eff[pk_idx]
            P_avg  = float(np.mean(P_in[P_in > 0])) if np.any(P_in > 0) else 0
            annual_cost = P_avg / 1000 * hrs * cost

            info = (
                f"Peak efficiency  = {pk_eff:.2f}% @ {pk_n:.0f} rpm\n"
                f"Avg input power  = {P_avg:.1f} W\n"
                f"Annual energy    = {P_avg/1000*hrs:.1f} kWh\n"
                f"Annual cost      = ${annual_cost:.2f}\n"
                f"\n(Pcore={Pcore:.0f}W, Pmech={Pmech:.0f}W)\n"
            )
            self._result9.config(state=tk.NORMAL)
            self._result9.delete("1.0", tk.END)
            self._result9.insert(tk.END, info)
            self._result9.config(state=tk.DISABLED)
        except Exception as e:
            self._result9.config(state=tk.NORMAL)
            self._result9.insert(tk.END, f"\nError: {e}\n")
            self._result9.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 10 – Harmonics and Power Quality
    # =======================================================================
    def _build_tab10(self):
        """Tab 10: FFT, THD, harmonic spectrum, IEEE 519 compliance."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 10 – Harmonics ")
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Harmonics & Power Quality – FFT / THD / IEEE 519",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        plot_frame = ttk.LabelFrame(frame, text="Harmonic Spectrum")
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._fig10 = Figure(figsize=(7, 5))
        self._ax10a, self._ax10b = self._fig10.subplots(2, 1)
        self._fig10.tight_layout(pad=2.5)
        self._canvas10, _ = embed_figure(self._fig10, plot_frame)

        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ttk.Label(ctrl, text="PWM Settings", font=("Helvetica", 10, "bold")).pack(pady=4)

        self._v_fpwm  = tk.DoubleVar(value=1000.0)
        self._v_mf    = tk.DoubleVar(value=50.0)
        self._v_ma    = tk.DoubleVar(value=0.9)
        self._v_nload = tk.DoubleVar(value=900.0)

        def _sl10(lbl, var, lo, hi, fmt):
            ttk.Label(ctrl, text=lbl).pack()
            lv = ttk.Label(ctrl, text=fmt.format(var.get()))
            lv.pack()
            def cb(val, lv=lv, f=fmt):
                lv.config(text=f.format(float(val)))
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, command=cb).pack(fill=tk.X, padx=6, pady=1)

        _sl10("PWM freq [Hz]:", self._v_fpwm, 100, 20000, "{:.0f} Hz")
        _sl10("Freq ratio mf:", self._v_mf,   10,  200,   "{:.0f}")
        _sl10("Mod index ma:",  self._v_ma,  0.1,   1.2,  "{:.2f}")
        _sl10("Speed [rpm]:",   self._v_nload, 10, 3000,  "{:.0f} rpm")

        ttk.Button(ctrl, text="Compute THD", command=self._update_tab10).pack(pady=8)
        self._result10 = scrolledtext.ScrolledText(ctrl, height=12, font=("Courier", 9))
        self._result10.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._update_tab10()

    def _update_tab10(self):
        try:
            vt, ra, n0, poles = self._params()
            fpwm   = clamp(float(self._v_fpwm.get()),  50, 100000)
            mf     = clamp(int(float(self._v_mf.get())),  2, 500)
            ma     = clamp(float(self._v_ma.get()),   0.01, 1.5)
            n_run  = clamp(float(self._v_nload.get()),  1, n0*2)
            Ke     = ke_rad(vt, n0)
            omega  = n_run * 2 * math.pi / 60
            Ea     = Ke * omega
            Ia_ss  = (vt - Ea) / ra

            # Synthesise approximate PWM armature current waveform
            fs   = fpwm * 20
            T    = 1.0 / (fpwm / mf)   # fundamental period
            t    = np.linspace(0, 5 * T, int(fs * 5 * T) + 1)
            f0   = fpwm / mf
            # Fundamental + PWM sidebands
            i_wfm = (Ia_ss * np.sin(2 * math.pi * f0 * t)
                     + 0.05 * Ia_ss * np.sin(2 * math.pi * fpwm * t)
                     + 0.03 * Ia_ss * np.sin(2 * math.pi * (fpwm + f0) * t)
                     + 0.02 * Ia_ss * np.sin(2 * math.pi * (fpwm - f0) * t)
                     + 0.15 * Ia_ss * np.sin(2 * math.pi * 3 * f0 * t)  # 3rd harmonic
                     + 0.07 * Ia_ss * np.sin(2 * math.pi * 5 * f0 * t)  # 5th
                     + 0.04 * Ia_ss * np.sin(2 * math.pi * 7 * f0 * t)  # 7th
                    )

            N = len(i_wfm)
            fft_vals = np.abs(np.fft.rfft(i_wfm)) / N * 2
            freqs    = np.fft.rfftfreq(N, 1.0 / fs)

            # Locate fundamental bin by expected frequency f0 (not argmax, avoids DC errors)
            df = fs / N  # FFT bin width [Hz]
            fund_bin = max(1, int(round(f0 / df)))
            fund_amp = fft_vals[fund_bin] if fund_bin < len(fft_vals) else 0.0

            # Extract harmonic amplitudes at 3rd, 5th, 7th, 9th harmonics (frequency windows)
            harm_amps = []
            for h_order in [3, 5, 7, 9]:
                h_bin = int(round(h_order * f0 / df))
                if h_bin < len(fft_vals):
                    # Peak within ±2 bins around expected harmonic
                    lo_b = max(0, h_bin - 2)
                    hi_b = min(len(fft_vals) - 1, h_bin + 2)
                    harm_amps.append(float(np.max(fft_vals[lo_b:hi_b + 1])))

            THD = 0.0
            if fund_amp > 1e-9:
                THD = math.sqrt(sum(h**2 for h in harm_amps)) / fund_amp * 100

            # Displacement power factor: PF = I1_rms / I_total_rms
            i_total_rms_sq = fund_amp**2 + sum(h**2 for h in harm_amps)
            pf = fund_amp / math.sqrt(i_total_rms_sq + 1e-9)

            # Plots
            ax1, ax2 = self._ax10a, self._ax10b
            ax1.cla(); ax2.cla()
            # Protect against f0 = 0 before computing plot slice length
            n_plot = int(3 / max(f0, 1e-6) * fs)
            n_plot = min(n_plot, len(t))
            ax1.plot(t[:n_plot] * 1000, i_wfm[:n_plot], color="C0", linewidth=0.8)
            ax1.set_ylabel("Ia (A)")
            ax1.set_xlabel("Time (ms)")
            ax1.set_title("Armature Current Waveform (3 cycles)")
            ax1.grid(True)

            # Spectrum up to 3× PWM freq
            mask = freqs <= fpwm * 3
            ax2.bar(freqs[mask], fft_vals[mask], width=freqs[1]-freqs[0]+1e-6,
                    color="C3", alpha=0.8)
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("|Ia(f)| (A)")
            ax2.set_title(f"Harmonic Spectrum (THD={THD:.1f}%)")
            ax2.grid(True)
            self._fig10.tight_layout(pad=2.5)
            self._canvas10.draw()

            # IEEE 519: THD < 5% at PCC
            ieee519 = "PASS ✓" if THD < 5.0 else "FAIL ✗"
            f_armature = poles * n_run / 120
            info = (
                f"Motor speed      = {n_run:.0f} rpm\n"
                f"Armature freq    = {f_armature:.2f} Hz\n"
                f"Ia (steady-state)= {Ia_ss:.2f} A\n"
                f"PWM frequency    = {fpwm:.0f} Hz\n"
                f"Fundamental freq = {f0:.1f} Hz\n"
                f"THD              = {THD:.2f}%\n"
                f"Power factor     ≈ {pf:.4f}\n"
                f"IEEE 519 (<5%)   = {ieee519}\n\n"
                f"Problem 29b:\n"
                f"  f at 300 rpm   = {poles*300/120:.1f} Hz\n"
                f"  f = P×n/120 = {poles}×300/120\n"
            )
            self._result10.config(state=tk.NORMAL)
            self._result10.delete("1.0", tk.END)
            self._result10.insert(tk.END, info)
            self._result10.config(state=tk.DISABLED)
        except Exception as e:
            self._result10.config(state=tk.NORMAL)
            self._result10.insert(tk.END, f"\nError: {e}\n")
            self._result10.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 11 – Comprehensive Dashboard
    # =======================================================================
    def _build_tab11(self):
        """Tab 11: All KPIs, efficiency map contour, operating envelope."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 11 – Dashboard ")
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Comprehensive Dashboard – KPIs & Efficiency Map",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")

        # KPI panel
        kpi_frame = ttk.LabelFrame(frame, text="Key Performance Indicators")
        kpi_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self._tab11_kpi = scrolledtext.ScrolledText(kpi_frame, font=("Courier", 10))
        self._tab11_kpi.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Efficiency map
        plot_frame = ttk.LabelFrame(frame, text="Efficiency Map (Torque vs Speed Contour)")
        plot_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self._fig11 = Figure(figsize=(6, 5))
        self._ax11 = self._fig11.add_subplot(111)
        self._canvas11, _ = embed_figure(self._fig11, plot_frame)

        ttk.Button(frame, text="Refresh Dashboard",
                   command=self._update_tab11).grid(row=2, column=0, columnspan=2, pady=6)

        self._update_tab11()

    def _update_tab11(self):
        try:
            vt, ra, n0, poles = self._params()
            Ke = ke_rad(vt, n0)
            Pcore = 50.0; Pmech = 30.0

            # ---- Efficiency map contour ------------------------------------
            n_arr = np.linspace(50, n0 * 1.05, 80)
            T_arr = np.linspace(0.1, vt / ra * Ke * 1.05, 80)
            N_grid, T_grid = np.meshgrid(n_arr, T_arr)
            omega_grid = N_grid * 2 * math.pi / 60
            Ia_grid    = T_grid / Ke
            Ea_grid    = Ke * omega_grid
            P_in_grid  = vt * np.maximum(Ia_grid, 0)
            P_cu_grid  = Ia_grid**2 * ra
            P_out_grid = T_grid * omega_grid
            eff_grid   = np.where(P_in_grid > 0,
                                  np.clip(P_out_grid / P_in_grid * 100, 0, 100),
                                  0.0)

            ax = self._ax11
            ax.cla()
            cf = ax.contourf(N_grid, T_grid, eff_grid,
                             levels=np.arange(0, 101, 5), cmap="RdYlGn")
            self._fig11.colorbar(cf, ax=ax, label="Efficiency (%)")

            # Operating envelope (max torque line)
            n_env = np.linspace(0, n0, 200)
            T_env = Ke * (vt - Ke * n_env * 2 * math.pi / 60) / ra
            T_env = np.maximum(T_env, 0)
            ax.plot(n_env, T_env, "w--", linewidth=2, label="Max torque envelope")

            # Problem 28 operating points
            pts = [(0, 76.39, "Start"), (300, 63.66, "28c"),
                   (900, 38.20, "28a"), (1200, 25.46, "28b")]
            for (nn, tt, lbl) in pts:
                ax.plot(nn, tt, "wo", markersize=8)
                ax.annotate(lbl, (nn, tt), textcoords="offset points",
                            xytext=(5, 5), color="white", fontsize=8)

            ax.set_xlabel("Speed (rpm)")
            ax.set_ylabel("Torque (N·m)")
            ax.set_title("Motor Efficiency Map")
            ax.legend(fontsize=8, loc="upper right")
            self._fig11.tight_layout()
            self._canvas11.draw()

            # KPI text
            Ia_fl = (vt - Ke * n0 * 2 * math.pi / 60 * 0.9) / ra
            P_out_fl = Ke * Ia_fl * n0 * 0.9 * 2 * math.pi / 60
            P_in_fl  = vt * Ia_fl
            eff_fl   = P_out_fl / P_in_fl * 100 if P_in_fl > 0 else 0

            kpi = (
                f"Motor: {poles}-pole Shunt DC, Vt={vt:.0f}V, Ra={ra:.2f}Ω\n"
                f"No-load speed n₀ = {n0:.0f} rpm\n"
                f"K_e (rad)        = {Ke:.4f} V·s/rad\n"
                f"──────────────────────────────────────\n"
                f"Starting Ia      = {vt/ra:.2f} A\n"
                f"Starting Torque  = {Ke*vt/ra:.2f} N·m\n"
                f"──────────────────────────────────────\n"
                f"Prob 28a (900rpm): Ia={30:.0f}A, T={38.20:.2f}N·m\n"
                f"Prob 28b (1200rpm): P={3200:.0f}W = {3200*HP_PER_WATT:.2f}hp\n"
                f"Prob 28c (300rpm): T={63.66:.2f}N·m\n"
                f"Prob 28d (start):  T={76.39:.2f}N·m = 56.35ft·lbf\n"
                f"──────────────────────────────────────\n"
                f"Prob 29a: n₀'(60V)= {60/(vt/n0):.1f} rpm\n"
                f"Prob 29b: f(300rpm)= {poles*300/120:.1f} Hz\n"
                f"──────────────────────────────────────\n"
                f"FL efficiency    ≈ {eff_fl:.1f}%\n"
                f"Annual cost @{0.15:.2f}$/kWh, 4000h/yr\n"
                f"  = ${P_in_fl/1000*4000*0.15:.0f}/yr\n"
            )
            self._tab11_kpi.config(state=tk.NORMAL)
            self._tab11_kpi.delete("1.0", tk.END)
            self._tab11_kpi.insert(tk.END, kpi)
            self._tab11_kpi.config(state=tk.DISABLED)
        except Exception as e:
            self._tab11_kpi.config(state=tk.NORMAL)
            self._tab11_kpi.insert(tk.END, f"\nError: {e}\n")
            self._tab11_kpi.config(state=tk.DISABLED)

    # =======================================================================
    # TAB 12 – Model Verification
    # =======================================================================
    def _build_tab12(self):
        """Tab 12: Calculated vs expected values with Pass/Fail."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 12 – Verify ")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Model Verification – Calculated vs Expected (Pass/Fail)",
                  font=("Helvetica", 12, "bold")).grid(row=0, column=0, columnspan=2,
                                                        pady=6, padx=8, sticky="w")
        # Tolerance slider
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=0, column=1, sticky="e", padx=8)
        ttk.Label(ctrl, text="Tolerance %:").pack(side=tk.LEFT)
        self._v_tol = tk.DoubleVar(value=1.0)
        lbl_tol = ttk.Label(ctrl, text="1.00 %")
        lbl_tol.pack(side=tk.RIGHT)
        def _tol_cb(val, lv=lbl_tol):
            lv.config(text=f"{float(val):.2f} %")
        ttk.Scale(ctrl, from_=0.01, to=10.0, variable=self._v_tol,
                  orient=tk.HORIZONTAL, command=_tol_cb, length=180).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Re-run", command=self._update_tab12).pack(side=tk.LEFT, padx=4)

        plot_frame = ttk.LabelFrame(frame, text="Verification Results")
        plot_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        self._fig12 = Figure(figsize=(10, 5))
        self._ax12a = self._fig12.add_subplot(121)
        self._ax12b = self._fig12.add_subplot(122)
        self._canvas12, _ = embed_figure(self._fig12, plot_frame)

        txt_frame = ttk.LabelFrame(frame, text="Verification Log")
        txt_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(2, weight=1)
        self._result12 = scrolledtext.ScrolledText(txt_frame, font=("Courier", 10), height=10)
        self._result12.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._update_tab12()

    def _update_tab12(self):
        try:
            vt, ra, n0, poles = self._params()
            Ke  = vt / n0                  # V/(r/min)
            Ker = ke_rad(vt, n0)           # V·s/rad
            tol = clamp(float(self._v_tol.get()), 1e-4, 100.0) / 100.0

            # Expected values (from textbook / problem statement)
            expected = {
                "Ia @ 900rpm (A)":    30.0,
                "P_mech @ 1200rpm (W)": 3200.0,
                "P_mech (hp)":        3200 * HP_PER_WATT,
                "Torque @ 300rpm (N·m)": 63.66,
                "Ia_start (A)":       60.0,
                "T_start (N·m)":      76.39,
                "T_start (ft·lbf)":   56.35,
                "n0_60V (rpm)":       450.0,
                "f_300rpm (Hz)":      10.0,
            }

            # Calculated values
            def calc_Ia(n):
                return (vt - Ke * n) / ra

            calculated = {
                "Ia @ 900rpm (A)":    calc_Ia(900),
                "P_mech @ 1200rpm (W)": (Ke * 1200) * calc_Ia(1200),
                "P_mech (hp)":        (Ke * 1200) * calc_Ia(1200) * HP_PER_WATT,
                "Torque @ 300rpm (N·m)": Ker * calc_Ia(300),
                "Ia_start (A)":       vt / ra,
                "T_start (N·m)":      Ker * (vt / ra),
                "T_start (ft·lbf)":   Ker * (vt / ra) * FTLBF_PER_NM,
                "n0_60V (rpm)":       60.0 / Ke,
                "f_300rpm (Hz)":      poles * 300 / 120,
            }

            labels = list(expected.keys())
            exp_vals  = np.array([expected[k]   for k in labels])
            calc_vals = np.array([calculated[k] for k in labels])
            errors    = np.abs(calc_vals - exp_vals) / (np.abs(exp_vals) + 1e-12) * 100
            passed    = errors <= tol * 100

            ax1, ax2 = self._ax12a, self._ax12b
            ax1.cla(); ax2.cla()

            short = [l.split("(")[0].strip() for l in labels]
            x     = np.arange(len(labels))
            w     = 0.35
            ax1.bar(x - w/2, exp_vals,  w, label="Expected", color="C0", alpha=0.8)
            ax1.bar(x + w/2, calc_vals, w, label="Calculated", color="C3", alpha=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
            ax1.set_title("Expected vs Calculated")
            ax1.legend(fontsize=8)
            ax1.grid(True, axis="y", alpha=0.4)

            colors_bar = ["green" if p else "red" for p in passed]
            ax2.bar(x, errors, color=colors_bar, alpha=0.8)
            ax2.axhline(tol * 100, color="k", linestyle="--",
                        label=f"Tol={tol*100:.2f}%")
            ax2.set_xticks(x)
            ax2.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
            ax2.set_title("Percentage Error")
            ax2.set_ylabel("Error (%)")
            ax2.legend(fontsize=8)
            ax2.grid(True, axis="y", alpha=0.4)
            self._fig12.tight_layout(pad=2.5)
            self._canvas12.draw()

            # Log
            lines = ["Parameter                      | Expected    | Calculated  | Error%  | Result\n"]
            lines.append("-" * 80 + "\n")
            for lbl, ev, cv, err, pas in zip(labels, exp_vals, calc_vals, errors, passed):
                status = "PASS ✓" if pas else "FAIL ✗"
                lines.append(f"{lbl:<32s}| {ev:10.4f}  | {cv:10.4f}  | {err:6.3f}% | {status}\n")
            n_pass = sum(passed)
            n_total = len(passed)
            lines.append("\n" + "=" * 80 + "\n")
            lines.append(f"OVERALL: {n_pass}/{n_total} tests PASSED  "
                         f"({'ALL OK ✓' if n_pass==n_total else 'SOME FAILED ✗'})\n")

            self._result12.config(state=tk.NORMAL)
            self._result12.delete("1.0", tk.END)
            for line in lines:
                self._result12.insert(tk.END, line)
            self._result12.config(state=tk.DISABLED)
        except Exception as e:
            self._result12.config(state=tk.NORMAL)
            self._result12.insert(tk.END, f"\nError: {e}\n")
            self._result12.config(state=tk.DISABLED)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    app = DCMotorApp()
    app.mainloop()
