"""Qblox broadband qubit spectroscopy — supplies only ``probe()``.

Parameters, fitting, simulation are inherited from
``scqo.experiments.BroadbandQubitSpectroscopy``.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import BroadbandQubitSpectroscopy

from ._broadband import read_field, stitch_subbands, write_field
from ._reset import add_reset
from ._vendor import vendor_element


@register
class QbloxBroadbandQubitSpectroscopy(BroadbandQubitSpectroscopy):
    """Build and execute wideband two-tone qubit spectroscopy across stepped drive LOs on Qblox."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run broadband_qubit_spectroscopy --help`
    backend_notes = (
        "one target per run: the probe steps one drive clock's local oscillator, and a "
        "second target is refused by name",
        "--preview is refused: the probe acquires band by band itself",
    )

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "broadband qubit spectroscopy steps drive LO frequencies across sub-bands"

    def build_sub_schedule(
        self,
        target: str,
        f_start: float,
        f_stop: float,
        n_pts: int,
        reps: int,
    ) -> Any:
        """Build a single-segment 1D frequency sweep schedule for qubit spectroscopy."""
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            SquarePulse,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        view = self.device.channel(target, "drive")
        element = vendor_element(self, target, "drive")
        drive_amp = float(view.drive_amp)
        drive_clock = f"{target}.01"
        mw_port = element.ports.microwave
        mw_port = mw_port() if callable(mw_port) else mw_port
        # / 1e9, never * 1e-9: the probes' float-exactness rule
        drive_len_s = round(float(self.params.drive_len_ns)) / 1e9

        schedule = Schedule(f"broadband_qubit_spec_{target}")
        with schedule.loop(arange(0, reps, 1, DType.NUMBER)):
            with schedule.loop(
                linspace(f_start, f_stop, n_pts, dtype=DType.FREQUENCY)
            ) as freq:
                schedule.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                add_reset(schedule, self, target)
                # A FINITE saturation pulse, over before the readout tone starts —
                # the same sequence the QM sibling plays (it reuses that backend's
                # qubit_spectroscopy builder verbatim). This used to be a latched
                # VoltageOffset held across the whole sub-band, which left the
                # drive live through every Measure. ASAP chaining is the anchor:
                # the pulse has a length, so the Measure lands at its end.
                schedule.add(
                    SquarePulse(drive_amp, drive_len_s, port=mw_port, clock=drive_clock)
                )
                schedule.add(
                    Measure(
                        target,
                        coords={f"frequency_{target}": freq},
                        acq_channel=f"S_21_{target}",
                    )
                )
                schedule.add(IdlePulse(4e-9))
        return schedule

    def probe(self) -> xr.Dataset:
        from scqo_qblox.backend.qblox_backend import chunk_timeout_s

        # ONE target only. Each qubit drives its own port-clock and this probe
        # steps exactly one of them (drive_port_clock below), so a second target
        # would never actually be swept -- reporting a row for it would mean
        # copying the first qubit's spectrum onto it. The QM backend multiplexes
        # an acquisition stream per target and CAN do several at once; refuse by
        # name here rather than silently downgrade.
        if len(self.params.targets) != 1:
            raise NotImplementedError(
                f"{self.name}: the Qblox backend steps ONE drive LO per run, so "
                f"it measures a single target; got {list(self.params.targets)}. "
                f"Run it once per qubit, or use the QM backend, which "
                f"multiplexes the sub-band acquisitions")
        primary_target = self.params.targets[0]
        chan_name = self.device.roster.default_channel(primary_target, "drive")
        view = self.backend.device.component(chan_name)
        drive_port_clock = getattr(view, "_drive_port_clock", getattr(view, "_port_clock", None))()

        hw_agent = self.backend._hw_agent
        hw_cfg = hw_agent.hardware_configuration
        opts = hw_cfg.hardware_options
        mf = getattr(opts, "modulation_frequencies", None)
        if mf is None:
            opts.modulation_frequencies = {}
            mf = opts.modulation_frequencies

        mf_entry = mf.get(drive_port_clock)
        if isinstance(mf_entry, dict):
            orig_lo = mf_entry.get("lo_freq")
        else:
            orig_lo = getattr(mf_entry, "lo_freq", None)

        # Save original drive clock frequencies for all targets
        orig_drive_clocks: dict[str, float] = {}
        for q_name in self.params.targets:
            ch_name = self.device.roster.default_channel(q_name, "drive")
            el = self.backend.device.component(ch_name)._element
            val = read_field(el.clock_freqs, "f01")
            if val is not None:
                orig_drive_clocks[q_name] = float(val)

        start = float(self.params.start_freq_hz)
        stop = float(self.params.stop_freq_hz)
        bw = float(self.params.bandwidth_per_lo_hz)
        pts_per_lo = int(self.params.num_points_per_lo)
        gap = float(self.params.lo_gap_hz)
        reps = int(self.params.num_averages)

        # Single-sideband IF offset avoiding LO leakage: IF in [min_if, max_if]
        min_if = max(20.0e6, gap / 2.0)
        max_if = min(400.0e6, min_if + bw)
        span_per_lo = max_if - min_if
        # Overlap margin between adjacent sub-bands for phase & gain alignment
        overlap = min(30.0e6, span_per_lo * 0.15) if (stop - start) > span_per_lo else 0.0

        slices: list[tuple[float, float, float]] = []
        curr_f = start
        while curr_f < stop:
            next_f = min(stop, curr_f + span_per_lo)
            lo = curr_f - min_if
            slices.append((curr_f, next_f, lo))
            if next_f >= stop:
                break
            curr_f = next_f - overlap

        subband_data: list[tuple[np.ndarray, np.ndarray]] = []

        try:
            for f_a, f_b, lo in slices:
                slice_span = f_b - f_a
                if slice_span <= 0:
                    continue

                n_pts = max(2, int(round(pts_per_lo * (slice_span / span_per_lo))))
                rf_seg = np.linspace(f_a, f_b, n_pts)

                # Update drive LO frequency in hardware config for this sub-band
                if isinstance(mf_entry, dict):
                    mf_entry["lo_freq"] = lo
                elif hasattr(mf_entry, "lo_freq"):
                    mf_entry.lo_freq = lo
                elif isinstance(mf, dict):
                    try:
                        from qblox_scheduler.backends.types.qblox import ModulationFrequencies

                        mf[drive_port_clock] = ModulationFrequencies(lo_freq=lo)
                    except Exception:
                        mf[drive_port_clock] = {"lo_freq": lo}

                # Synchronize base drive clock frequency to f_a (initial NCO = f_a - lo = min_if)
                # NOT best-effort: a swallowed failure here measures the
                # whole sub-band at the PREVIOUS LO's frequency and labels it
                # with this one's.
                for q_name in orig_drive_clocks:
                    ch_name = self.device.roster.default_channel(q_name, "drive")
                    el = self.backend.device.component(ch_name)._element
                    write_field(el.clock_freqs, "f01", f_a)

                sub = self.build_sub_schedule(primary_target, f_a, f_b, n_pts, reps)
                qd = hw_agent.quantum_device
                qd.hardware_config = hw_cfg

                timeout_s = chunk_timeout_s(self, shots=reps, points=n_pts)
                hw_agent.instrument_coordinator.timeout(timeout_s)
                raw_sub = hw_agent.run(sub, timeout=timeout_s)

                key = f"S_21_{primary_target}"
                if key in raw_sub.data_vars:
                    vals = np.asarray(raw_sub[key].values).squeeze()
                    subband_data.append((rf_seg, vals))
        finally:
            # Restore original drive LO setting
            if isinstance(mf_entry, dict):
                mf_entry["lo_freq"] = orig_lo
            elif hasattr(mf_entry, "lo_freq"):
                mf_entry.lo_freq = orig_lo
            elif isinstance(mf, dict):
                if orig_lo is not None:
                    try:
                        from qblox_scheduler.backends.types.qblox import ModulationFrequencies

                        mf[drive_port_clock] = ModulationFrequencies(lo_freq=orig_lo)
                    except Exception:
                        mf[drive_port_clock] = {"lo_freq": orig_lo}
                else:
                    # There was no entry before this probe injected one; leaving
                    # it behind would pin a synthetic LO override on the port for
                    # the rest of the session.
                    mf.pop(drive_port_clock, None)

            # Restore original drive clock frequencies
            for q_name, orig_clk in orig_drive_clocks.items():
                try:
                    ch_name = self.device.roster.default_channel(q_name, "drive")
                    el = self.backend.device.component(ch_name)._element
                    write_field(el.clock_freqs, "f01", orig_clk)
                except Exception:
                    pass

        final_freqs, final_z = stitch_subbands(subband_data)

        # exactly one target, enforced at the top of probe()
        targets = [primary_target]
        i_2d = np.real(final_z)[np.newaxis, :]
        q_2d = np.imag(final_z)[np.newaxis, :]

        return xr.Dataset(
            data_vars={
                "I": (("target", "frequency_hz"), i_2d),
                "Q": (("target", "frequency_hz"), q_2d),
            },
            coords={
                "target": targets,
                "frequency_hz": final_freqs,
            },
        )
