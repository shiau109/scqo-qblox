"""Qblox DRAG Equator (3-Line) calibration acquisition probe.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitDragEquator``.
"""

from __future__ import annotations

from typing import Any, ClassVar
import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import QubitDragEquator
from scqo.requirements import Requirement

from ._broadband import read_field, write_field
from ._requires import HALF_PI_AMPLITUDE
from ._reset import add_reset
from ._state import measure_kwargs


@register
class QbloxQubitDragEquator(QubitDragEquator):
    """Build and execute the DRAG equator calibration across swept beta on Qblox."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitDragEquator.requires, HALF_PI_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in `scqo run qubit_drag_equator --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "each coefficient is realized by writing drag_beta on the device, one program per "
        "point, and the stored value is put back afterwards; --preview is refused",
        "one DRAG coefficient serves every rotation here: target_gate=x90 plays the doubled "
        "pi/2 sequence but still sweeps drag_beta, and its proposal drag_beta_x90 is not "
        "realized on this backend",
        "the pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
    )

    probe_self_acquires = (
        "it sweeps DRAG beta by stepping element.rxy.beta and acquiring across schedules"
    )

    def build_schedule(self, label: str = "drag_equator") -> Any:
        """The two-sequence schedule for the beta currently on the device.

        probe() steps ``element.rxy.beta`` between calls -- the beta is device
        state, not a schedule parameter, so ONE call is the whole per-point
        program. Split out so tests/test_probe_surface.py can compile it: a
        self-acquiring probe otherwise gets no compile coverage at all, and the
        compiler is where the time grid and the DAC range are enforced.
        """
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange

        qubits = list(self.params.targets)
        reps = int(self.params.num_averages)
        target_gate = getattr(self.params, "target_gate", "x180")
        op_name = "x90" if str(target_gate).strip().lower() == "x90" else "x180"

        schedule = Schedule(label)
        for q_name in qubits:
            acq = measure_kwargs(self, q_name)
            sub = Schedule(f"drag_equator_{q_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                # Seq 0: Rx(pi) - Ry(pi/2)  (or two x90 - Ry(pi/2))
                add_reset(sub, self, q_name)
                if op_name == "x90":
                    sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                    sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                    sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                else:
                    sub.add(Rxy(theta=180.0, phi=0.0, qubit=q_name))
                    sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                sub.add(
                    Measure(
                        q_name,
                        coords={f"seq_{q_name}": 0},
                        acq_channel=f"S_21_{q_name}",
                        **acq,
                    )
                )
                sub.add(IdlePulse(4e-9))

                # Seq 1: Ry(pi) - Rx(pi/2)  (or two y90 - Rx(pi/2))
                add_reset(sub, self, q_name)
                if op_name == "x90":
                    sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                    sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
                    sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                else:
                    sub.add(Rxy(theta=180.0, phi=90.0, qubit=q_name))
                    sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
                sub.add(
                    Measure(
                        q_name,
                        coords={f"seq_{q_name}": 1},
                        acq_channel=f"S_21_{q_name}",
                        **acq,
                    )
                )
                sub.add(IdlePulse(4e-9))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule

    def probe(self) -> Any:
        from scqo_qblox.backend.qblox_backend import chunk_timeout_s

        qubits = list(self.params.targets)
        beta_array = np.asarray(self.sweep_axes["beta"], dtype=float)
        reps = int(self.params.num_averages)

        n_qubits = len(qubits)
        n_seq = 2
        n_beta = len(beta_array)

        i_data = np.zeros((n_qubits, n_seq, n_beta), dtype=float)
        q_data = np.zeros((n_qubits, n_seq, n_beta), dtype=float)

        # What to put back afterwards. `or 0.0` would fold an UNSET beta into
        # a real 0.0 and then write it, quietly discarding whatever DRAG the
        # chip had; leave unset qubits out of the restore map instead.
        orig_betas: dict[str, float] = {}
        for q_name in qubits:
            stored = self.device.channel(q_name, "drive").drag_beta
            if stored is not None:
                orig_betas[q_name] = float(stored)

        hw_agent = self.backend._hw_agent

        try:
            for i_b, beta_val in enumerate(beta_array):
                # Update DRAG beta on each qubit channel before building the schedule
                for q_name in qubits:
                    self.device.channel(q_name, "drive").drag_beta = float(beta_val)

                schedule = self.build_schedule(f"drag_equator_beta_{i_b}")

                # two sequences per shot, one program per beta point
                timeout_s = chunk_timeout_s(self, shots=reps, points=2)
                hw_agent.instrument_coordinator.timeout(timeout_s)
                raw = hw_agent.run(schedule, timeout=timeout_s)

                for k, q_name in enumerate(qubits):
                    key = f"S_21_{q_name}"
                    if key not in raw.data_vars:
                        raise KeyError(f"acquisition channel {key!r} not in raw dataset")
                    values = np.asarray(raw[key].values).squeeze()
                    # values has 2 points corresponding to seq 0 and seq 1
                    i_data[k, 0, i_b] = float(np.real(values[0]))
                    i_data[k, 1, i_b] = float(np.real(values[1]))
                    q_data[k, 0, i_b] = float(np.imag(values[0]))
                    q_data[k, 1, i_b] = float(np.imag(values[1]))
        finally:
            for q_name, orig_beta in orig_betas.items():
                self.device.channel(q_name, "drive").drag_beta = orig_beta

        coords = {
            "target": qubits,
            "seq_idx": np.array([0, 1]),
            "beta": beta_array,
        }
        dims = ("target", "seq_idx", "beta")
        return xr.Dataset(
            {"I": (dims, i_data), "Q": (dims, q_data)},
            coords=coords,
        )
