"""Declarative field catalog for the Qblox backend — PURE DATA, no vendor imports.

Keyed by CHANNEL KIND (``drive`` / ``readout`` / ``flux``) since the greenfield
model: knobs live on channel entities (``q1_xy``, ``q1_ro``, ``q1_z``), not on a
single per-qubit component. Per kind, one :class:`scqo.fieldmap.VendorBinding`
per realized KNOB (where it lives on the qblox_scheduler device tree, in what
unit, converted how — as a DESCRIPTION), one :class:`scqo.fieldmap.Unrealized`
per knob this backend cannot realize, plus the
:class:`scqo.fieldmap.VendorOnly` inventory of calibration-relevant knobs that
have no neutral counterpart yet. The EXECUTABLE conversions live in the three
channel views of ``qblox_backend.py`` (``QbloxDriveChannel`` /
``QbloxReadoutChannel`` / ``QbloxFluxChannel``) — this module documents them and
is pinned to the implementation by ``tests/test_scqo_glue.py``
(bindings | unrealized == scqo's KNOB fields per kind; imports stay vendor-free).

MONITORS are absent by construction: ``fidelity_g``/``fidelity_e``/``pos_*`` are
measured performance OF the current knobs, never pushed, so they need no vendor
binding and no Unrealized entry (the old ``readout_fidelity`` aggregate is gone;
the per-state pair replaces it). Facts (``flux_offset``, ``flux_per_phi0``,
``distortion_*``) live in physical.json and likewise bind nothing.

Rendered by ``scqo state --fields``; strings reach lab consoles, keep them ASCII.
"""

from __future__ import annotations

from scqo.fieldmap import OperatorCommand, Unrealized, VendorBinding, VendorOnly

FIELD_BINDINGS: dict[str, dict[str, VendorBinding]] = {
    "drive": {
        "drive_freq_hz": VendorBinding(
            path="element.clock_freqs.f01", unit="Hz"),
        "pi_amp": VendorBinding(
            path="element.rxy.amp180", unit=""),
        "pi_duration_s": VendorBinding(
            path="element.rxy.duration", unit="s",
            note="pi/x180 pulse length, seconds (chipA: 200 ns here vs 32 ns on "
                 "QM - genuinely per-chain calibrated). No 4 ns grid guard: the "
                 "portable-grid contract is the READOUT pulse/window pair, and "
                 "inventing one here would refuse values the scheduler accepts. "
                 "QM counterpart: xy.operations['x180'].length (ns)"),
        "thermalization_time_s": VendorBinding(
            path="element.reset.duration", unit="s",
            note="passive-reset wait between shots: the IdlePulse length the "
                 "scheduler's Reset() gate compiles to, so every probe honours "
                 "it with no probe-side argument. Absolute seconds on both "
                 "sides - the one field that maps 1:1. Calibrated by "
                 "qubit_relaxation (thermalization_factor x T1). QM "
                 "counterpart: q.thermalization_time_ns (ns, 4 ns grid), which "
                 "overrides QUAM's derived thermalization_time_factor * T1"),
        "drag_beta": VendorBinding(
            path="element.rxy.beta", unit="ns",
            convert="rxy.beta is SECONDS of derivative scaling; the view speaks "
                    "ns (beta_s = drag_beta_ns * 1e-9) so the knob lands on the "
                    "same numeric range as the neutral min_beta/max_beta window",
            note="the DRAG coefficient of the SHARED rxy op, calibrated by "
                 "qubit_drag_equator. Deliberately NOT the same quantity as the "
                 "QM side's DragCosinePulse.alpha -- catalog marks drag_beta "
                 "portable=False precisely so each backend may define it in its "
                 "own convention. rxy has ONE beta, so x180 and x90 share it: "
                 "see drag_beta_x90 in UNREALIZED"),
        "drive_amp": VendorBinding(
            path="element.spec.spec_amp", unit="",
            note="the saturation (spec) drive amplitude - the CW VoltageOffset "
                 "fraction the qubit_spectroscopy probe plays, and the "
                 "drive_power_dbm chain solve's residual (scqo_qblox.elements "
                 "LCHTransmonElement's spec submodule)"),
        "drive_power_dbm": VendorBinding(
            path='hardware_options.output_att["<mw-port>-<qubit>.01"] '
                 "+ element.spec.spec_amp",
            unit="dB + amp",
            convert="solve the DRIVE chain: largest EVEN output_att in [0, 60] keeping "
                    "spec_amp <= 0.5 against the nominal +5 dBm full scale; spec_amp "
                    "absorbs the exact residual (absolute power good to +/- a few dB)",
            coupled=("drive_amp",),
            note="the drive output_att is PORT-level and shared by every xy pulse: "
                 "while it is off its standing value the stored pi_amp means a "
                 "different power (qubit_spectroscopy sets it and reverts exactly); "
                 "EVEN integers 0-60 dB, hardware compilation config authoritative",
        ),
    },
    "readout": {
        "readout_freq_hz": VendorBinding(
            path="element.clock_freqs.readout", unit="Hz"),
        "readout_amp": VendorBinding(
            path="element.measure.pulse_amp", unit=""),
        "readout_power_dbm": VendorBinding(
            path='hardware_options.output_att["<ro-port>-<qubit>.ro"] '
                 "+ element.measure.pulse_amp",
            unit="dB + amp",
            convert="solve the output chain: largest EVEN output_att in [0, 60] keeping "
                    "pulse_amp <= 0.5 against the nominal +5 dBm full scale; pulse_amp "
                    "absorbs the exact residual (absolute power good to +/- a few dB)",
            coupled=("readout_amp",),
            note="output_att takes EVEN integers 0-60 dB (module validator); the "
                 "hardware compilation config is authoritative - recompiled and "
                 "re-pushed every run",
        ),
        "readout_duration_s": VendorBinding(
            path="element.measure.pulse_duration", unit="s",
            coupled=("readout_integration_s",),
            note="positive multiples of 4 ns only (the portable grid, from QM's "
                 "weights resolution; REFUSED otherwise, no silent rounding); "
                 "shrinking the pulse clamps integration_time down with it "
                 "(QM parity - its weights cannot span past the pulse)",
        ),
        "readout_integration_s": VendorBinding(
            path="element.measure.integration_time", unit="s",
            note="contract: <= readout_duration_s - the hardware here allows a "
                 "longer window but QM cannot realize one, so the portable "
                 "range stops at the pulse; multiples of 4 ns. QM counterpart: "
                 "zero-padded constant integration weights",
        ),
        "readout_depletion_s": VendorBinding(
            path="element.depletion.duration", unit="s",
            convert="none - absolute seconds on both sides, ROUNDED onto the "
                    "1 ns scheduler grid (a policy wait derived from a fit, "
                    "like thermalization_time_s; not refused, or the loop could "
                    "not write its own result)",
            note="the post-readout photon-depletion wait. Calibrated by "
                 "resonator_spectroscopy as depletion_factor / (2 pi x "
                 "kappa_tot_hz) - the readout twin of thermalization_time_s. "
                 "The slot is a LAB addition to the element (elements.py "
                 "DepletionProperties); NaN means never calibrated and active "
                 "reset refuses it. QM counterpart: q.resonator.depletion_time "
                 "(ns, 4 ns grid)"),
        "readout_rotation_rad": VendorBinding(
            path="element.measure.acq_rotation", unit="deg",
            convert="RADIANS -> DEGREES and NEGATED (the vendor knob is degrees "
                    "and opposite-handed: acq_rotation turns the data "
                    "counterclockwise - 'threshold line clockwise' per the vendor "
                    "tutorial - while the neutral field keeps QM's "
                    "integration_weights_angle convention, which turns it "
                    "clockwise). The ABSOLUTE demod rotation, not a delta: "
                    "single_shot_readout proposes -measured_delta - so a direct "
                    "edit silently de-calibrates it; governed write: "
                    "scqo set QUBIT.readout_rotation_rad=...",
            note="acquisition IQ frame, chain-dependent (invalidated by an "
                 "input-chain change such as readout_input_att); non-portable. "
                 "Read by the compiler ONLY for acq_protocol='ThresholdedAcquisition' "
                 "- the use_state_discrimination path",
        ),
        "readout_threshold": VendorBinding(
            path="element.measure.acq_threshold", unit="",
            convert="none - the threshold is compared against the rotated result in "
                    "the SAME normalized frame the probes acquire in (the compiler "
                    "multiplies by integration_length in ns; an SSBIntegrationComplex "
                    "result is divided by acq_duration), so a threshold solved from "
                    "acquired I/Q is directly usable. QM counterpart: readout "
                    "threshold in RAW DEMOD units - a different frame, which is why "
                    "this field is non-portable",
            note="the g/e cut use_state_discrimination applies on the FPGA; "
                 "acquisition-frame, chain-dependent. Vendor default 0.0 means "
                 "UNCALIBRATED - the probes refuse to discriminate on it",
        ),
    },
    "flux": {
        "idle_flux": VendorBinding(
            path="element.flux_params.sweet_spot", unit="V",
            note="the standing bias a RELATIVE-frame flux sweep is measured FROM: "
                 "the probes emit VoltageOffset(idle_flux) and play the excursion "
                 "on top, the same shape as QM's initialize_qpu + play('const'). "
                 "Was Unrealized until 2026-07-30 while the probes read this exact "
                 "field through a vendor helper - the home existed, the neutral "
                 "surface was routed around it. NaN means uncalibrated and REFUSES: "
                 "0.0 is where the absolute and relative frames coincide, so "
                 "defaulting to it would hide a wrong-frame sweep. QM counterpart: "
                 "q.z.<flux_point>_offset"),
    },
}

#: Neutral KNOBS this backend cannot realize (declared, never silent): pushes are
#: skipped with the reason visible to doctor and the catalog view. The scqo
#: dataclass attribute is still spelled ``category``; it carries the channel KIND.
UNREALIZED: dict[str, dict[str, Unrealized]] = {
    "drive": {
        "drag_beta_x90": Unrealized(
            "drive", "drag_beta_x90",
            "rxy carries a SINGLE beta, which drag_beta above now realizes, so an "
            "independent pi/2 DRAG has nowhere of its own to live -- binding it to "
            "rxy.beta would make an x90 calibration silently overwrite the x180 "
            "one. It lands with pi_amp_x90 below, on the same lab element field"),
        "pi_amp_x90": Unrealized(
            "drive", "pi_amp_x90",
            "the slot EXISTS -- PiHalfProperties.amp90 in scqo_qblox/elements.py -- "
            "but nothing PLAYS it: Qblox DERIVES X90 from rxy.amp180 as "
            "amp180*theta/180, so binding pi_amp_x90 to rxy.amp180 (as "
            "amp180 = 2*pi_amp_x90) would let an x90 calibration overwrite the "
            "calibrated pi amplitude. qubit_deterministic_benchmarking is realized "
            "here now but REFUSES target_gate=x90 by name for exactly this reason. "
            "Promoting means moving pi_half from FluxTunableTransmonElement up to "
            "the LCHTransmonElement base (fixed-frequency chips need it too), "
            "binding element.pi_half.amp90, and teaching the probes to PLAY it"),
    },
    "readout": {
        # rotation + threshold are REALIZED above (acq_rotation/acq_threshold);
        # only the RUS exit has no Qblox counterpart.
        "readout_rus_threshold": Unrealized(
            "readout", "readout_rus_threshold",
            "no SECOND threshold to write: this is a repeat-until-success LOOP "
            "EXIT, and Qblox active reset is not a loop. reset_method='active' "
            "here plays a fixed number of ConditionalResets (see "
            "experiments/_reset.py), each branching on acq_threshold, so there is "
            "nothing an exit threshold would control. NOT 'Qblox has no "
            "feedback' - it does, via conditional playback. QM's while_(I > "
            "rus_exit_threshold) is what this field exists for"),
    },
    "flux": {
        "flux_delay_s": Unrealized(
            "flux", "flux_delay_s",
            "no flux-line delay wired for Qblox yet: the hardware home exists — "
            "hardware_options.latency_corrections[<port-clock>] (seconds, applied "
            "per port-clock and min-subtracted across the cluster at compile) — "
            "but no scqo experiment writes it here (qubit_xyz_delay is QM-only "
            "today). Promote to a real latency_corrections binding when a Qblox "
            "XY-Z delay probe lands. idle_flux, the kind's other knob, is REALIZED "
            "above (element.flux_params.sweet_spot)"),
    },
}

#: Backend-unique calibration knobs, vendor-owned and untracked by SCQO (edit in
#: dut_config.json / hw_config.json). Each entry carries its placement-rule kind
#: (scqo state --rule): realizer / candidate / vendor / unique. Doubles as the
#: neutral-field promotion backlog (the candidates pre-declare their convention).
VENDOR_ONLY: dict[str, VendorOnly] = {
    "readout_pulse_duration": VendorOnly(
        path="element.measure.pulse_duration", unit="s", kind="realizer",
        doc="readout pulse length - realizes the TRACKED readout_duration_s",
        edit="scqo set QUBIT.readout_duration_s=... - a direct edit silently "
             "de-calibrates it",
        counterpart="readout.length (ns)"),
    "readout_integration_time": VendorOnly(
        path="element.measure.integration_time", unit="s", kind="realizer",
        doc="acquisition integration window - realizes the TRACKED "
            "readout_integration_s",
        edit="scqo set QUBIT.readout_integration_s=... (contract: window <= "
             "pulse, for QM portability)",
        counterpart="the integration-weights support"),
    "readout_acq_delay": VendorOnly(
        path="element.measure.acq_delay", unit="s", kind="vendor",
        doc="delay from readout pulse start to acquisition start - aligns the "
            "instrument's receive path with its own transmit path (cable+"
            "electronics latency). The TOF measurement's product is written "
            "HERE, in SECONDS, offline - never a neutral field",
        counterpart="resonator.time_of_flight (ns)"),
    "readout_lo_freq": VendorOnly(
        path='hardware_options.modulation_frequencies["<ro-port>-<qubit>.ro"].lo_freq',
        unit="Hz", kind="vendor",
        doc="readout LO - PORT-level wiring shared by every element on that "
            "output; many LO/IF splits give the SAME RF, so SCQO owns only the "
            "RF (readout_freq_hz) and never moves the LO in a chain solve",
        edit="hw_config.json hardware_options.modulation_frequencies, with NO "
             "session live (a session's save() rewrites the file from memory "
             "and would silently revert you) - restart notebook kernels after. "
             "Keep IF = readout_freq_hz - lo_freq in the sequencer NCO range",
        counterpart="opx_output.upconverter_frequency"),
    "drive_lo_freq": VendorOnly(
        path='hardware_options.modulation_frequencies["<mw-port>-<qubit>.01"].lo_freq',
        unit="Hz", kind="vendor",
        doc="drive LO - PORT-level, shared by every element on that output",
        edit="hw_config.json hardware_options.modulation_frequencies, with NO "
             "session live (a session's save() rewrites the file from memory "
             "and would silently revert you) - restart notebook kernels after. "
             "Keep IF = f01 - lo_freq in NCO range"),
    "output_att": VendorOnly(
        path='hardware_options.output_att["<ro-port>-<qubit>.ro"]',
        unit="dB", kind="realizer",
        doc="the coarse readout power knob (EVEN integers 0-60) - it REALIZES "
            "the tracked readout_power_dbm (binding above)",
        edit="scqo set QUBIT.readout_power_dbm=... (solves the chain, keeps "
             "readout_amp coupled, recorded); a direct edit silently "
             "de-calibrates the absolute power. To FORCE a value, hand-edit "
             "hw_config.json with no session live - but any later "
             "readout_power_dbm write re-solves and overwrites it"),
    "drive_output_att": VendorOnly(
        path='hardware_options.output_att["<mw-port>-<qubit>.01"]',
        unit="dB", kind="realizer",
        doc="the coarse DRIVE power knob (EVEN integers 0-60, PORT-level - "
            "shared by every xy pulse) - it REALIZES the tracked "
            "drive_power_dbm (binding above)",
        edit="scqo set QUBIT.drive_power_dbm=... (solves the chain, keeps "
             "drive_amp coupled, recorded); a direct edit silently re-scales "
             "what every stored pi_amp AND the absolute drive power mean. A "
             "forced value goes in hw_config.json with no session live, same "
             "rule as its readout twin in the same hardware_options block",
        counterpart="xy opx_output full_scale_power_dbm"),
    "x180_duration": VendorOnly(
        path="element.rxy.duration", unit="s", kind="realizer",
        doc="pi/x180 pulse length - it REALIZES the tracked pi_duration_s "
            "(promoted to a neutral drive knob in the greenfield catalog; "
            "binding above)",
        edit="scqo set QUBIT.pi_duration_s=... - a direct edit silently "
             "de-calibrates the stored pi_amp with it",
        counterpart="xy.operations['x180'].length (ns)"),
    # NOTE: drag_beta is a neutral DRIVE knob (QM-realized). It is Unrealized on
    # Qblox for now (see UNREALIZED above) - element.rxy.beta is the vendor knob
    # that WOULD realize it; the binding lands when a Qblox DRAG experiment is
    # written. Kept out of VENDOR_ONLY to avoid colliding with the tracked name.
    # NOTE: acq_threshold / acq_rotation are no longer VENDOR_ONLY - they REALIZE
    # the tracked readout_threshold / readout_rotation_rad (bindings above). A
    # direct edit silently de-calibrates the discriminator; the governed writes are
    # scqo set QUBIT.readout_threshold=... / .readout_rotation_rad=..., or a
    # single_shot_readout run + scqo accept.
    "readout_input_att": VendorOnly(
        path='hardware_options.input_att["<ro-port>-<qubit>.ro"]',
        unit="dB", kind="vendor",
        doc="acquisition input attenuation (chipA: 10 dB; input_gain gain_I/"
            "gain_Q sit beside it)",
        coupled=("readout_threshold", "readout_rotation_rad"),
        edit="an edit silently invalidates every discrimination value and all "
             "fidelity comparisons - re-run single_shot_readout after it",
        counterpart="mw_input gain_db"),
    "reference_magnitude": VendorOnly(
        path="element.measure.reference_magnitude", unit="dBm/V/A", kind="unique",
        doc="hardware-referenced amplitude scaling of the measure pulse - no QM "
            "counterpart: an experiment depending on it runs ONLY on Qblox"),
}

#: The vendor OPERATOR CLIs this driver ships. They are not scqo subcommands
#: (scqo run <name> is the single entry point, and a Qblox-specific verb could
#: only be refused on QM), so `scqo -h` cannot list them - `scqo state --fields`
#: renders this inventory instead, which is the only place an operator discovers
#: them rather than memorizing them. Declared HERE and not in qblox_backend.py
#: because it is pure declarative vendor metadata of the same class as
#: VENDOR_ONLY, and this module's import guard is what proves it stays
#: vendor-free. Every string below compresses the target module's own docstring.
OPERATOR_COMMANDS: tuple[OperatorCommand, ...] = (
    OperatorCommand(
        name="apply_distortion",
        command="python -m scqo_qblox.backend.apply_distortion --target <target> "
                "[--run <run_id>]",
        doc="Write accepted cryoscope taps (distortion_amp / distortion_tau_s "
            "are FACTS - accepting them records the measurement and pushes "
            "NOTHING) into ONE QbloxHardwareDistortionCorrection under "
            "hardware_options.distortion_corrections, saved to both config "
            "files and compiled + pushed on the next run that plays flux. Run "
            "it after a cryoscope run's facts are accepted; it is the same "
            "command the cryoscope writeback hint prints for you.",
        options="--run RUN_ID (taps from that run's fit - the iteration door)  "
                "--extend (merge + re-partition the bank, not append)  "
                "--clear (fresh-line reset)  --dry-run  --config PATH",
        caution="The 4-stage bank is hard hardware: overflow taps are DROPPED "
                "(the most significant by |A| are kept) and the run warns "
                "loudly. Read that warning before trusting the filter."),
    OperatorCommand(
        name="calibrate_mixers",
        command="python scripts/calibrate_mixers.py <config_dir>",
        doc="Suppress LO leakage and the image sideband on the RF modules using "
            "their built-in AMC - no spectrum analyzer, no cabling. An "
            "OPERATIONS tool, not part of the scqo surface: it drives "
            "qblox_instruments directly. Path-invoked from a scqo-qblox "
            "checkout (not a module, not a console script); <config_dir> is the "
            "setup's backend_config folder, which scqo doctor prints.",
        options="--attempts N (default 12 - THE reliability lever: sideband_cal "
                "returns having changed nothing on roughly half to "
                "three-quarters of calls)  --slot N  --port-clock  "
                "--force (ignore the cache)  --dry-run (plan only, no cluster)  "
                "--diagnose  --dummy",
        caution="no scqo session / HardwareAgent holding this cluster may be "
                "live while it runs, and output switches go OFF during "
                "calibration. It also ends its own process hard (os._exit) by "
                "design, so never import it into a live session."),
)
