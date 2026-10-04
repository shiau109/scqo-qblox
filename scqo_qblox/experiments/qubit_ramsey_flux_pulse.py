"""Qblox Ramsey vs flux PULSE — supplies only ``probe()``.

Per shot: reset -> Y90 at the idle point -> buffer -> the flux line held at
idle + a for the whole idle tau (a sticky ``VoltageOffset`` there and back, the
same emitted level as QM's ``play("const")`` riding on the standing bias) ->
buffer -> the closing pi/2 with the virtual phase ramp -> readout at the idle
point. Loops: averages (outer) -> flux amplitude (start -> end, as given) -> tau.

Parameters, the folding-safe SIGN of the virtual detuning, the fit and the park
writeback are inherited from ``scqo.experiments.QubitRamseyFluxPulse``; the signed
detuning arrives per target from ``ramp_detuning_hz``.

Sequence parity with scqo-qm: the opening pi/2 is Y90 (``Rxy(90, phi=90)``) and the
closing one ``Rxy(90, phi=-360*D*tau)`` - the same relative phase QM builds from
y90 + frame rotation + x90. The ramp sign is qubit_ramsey's (see that probe for
why it is negative and why it must be ``Rxy(phi=...)``, never ``X90(phase=...)``).

NOT HARDWARE-VALIDATED on Qblox: active reset stays refused (this repo's
convention for an unvalidated probe) and a foreign ``flux_component`` is refused
by name - the QM probe realizes both.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitRamseyFluxPulse
from scqo.experiments._capabilities import flux_anchor_v
from scqo.requirements import Requirement

from ._flux_limits import check_flux_pulse_relative, to_dac_fraction
from ._requires import HALF_PI_AMPLITUDE
from ._reset import add_reset
from ._state import measure_kwargs
from ._vendor import vendor_element


@register
class QbloxQubitRamseyFluxPulse(QubitRamseyFluxPulse):
    """Build a multiplexed Ramsey-vs-flux-pulse Schedule for a Qblox cluster."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitRamseyFluxPulse.requires, HALF_PI_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_ramsey_flux_pulse --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
        "the flux pulse is a voltage offset set for the idle and returned to the idle flux "
        "afterwards, not a shaped pulse",
        "the virtual detuning is the phase of the second Rxy; the same phase passed to X90 "
        "would be dropped by the compiler",
        "flux_component is refused: only the target's own flux line can be pulsed",
    )

    def probe(self) -> Any:
        if self.params.flux_component is not None:
            raise NotImplementedError(
                "flux_component is not realized on the Qblox backend yet: this "
                "probe pulses each target's OWN flux line only (an assigned source "
                "would be silently wrong, so it refuses)")
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy, VoltageOffset
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        axes = self.sweep_axes or self.define_sweep()
        flux_v = np.asarray(axes["flux_bias_v"], dtype=float)
        idle_ns = np.asarray(axes["idle_time_ns"], dtype=float)
        buffer_s = self.params.flux_buffer_ns * 1e-9
        reps = self.params.num_averages

        schedule = Schedule("qubit_ramsey_flux_pulse_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)
            flux_port = vendor_element(self, qubit_name, "flux").ports.flux
            detuning = self.ramp_detuning_hz(qubit_name)
            # the relative frame's origin, read exactly as estimate() records it
            idle_flux = flux_anchor_v(self, qubit_name)
            rail = check_flux_pulse_relative(
                self, name=f"{qubit_name} flux", port=flux_port,
                idle_v=idle_flux, amps_v=flux_v)
            sub = Schedule(f"ramsey_flux_pulse_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                # the DOMAIN carries idle + excursion (an arithmetic expression on a
                # loop variable does not compile - see qubit_spectroscopy_flux_pulse);
                # start -> end is the traversal order, either direction
                with sub.loop(
                    linspace(to_dac_fraction(idle_flux + float(flux_v[0]), rail),
                             to_dac_fraction(idle_flux + float(flux_v[-1]), rail),
                             flux_v.size, dtype=DType.AMPLITUDE)
                ) as flux:
                    with sub.loop(
                        linspace(idle_ns[0] * 1e-9, idle_ns[-1] * 1e-9, idle_ns.size,
                                 dtype=DType.TIME)
                    ) as tau:
                        add_reset(sub, self, qubit_name)
                        sub.add(Rxy(theta=90.0, phi=90.0, qubit=qubit_name))  # Y90
                        if buffer_s:
                            sub.add(IdlePulse(buffer_s))
                        sub.add(VoltageOffset(flux, 0, port=flux_port))
                        sub.add(IdlePulse(tau))
                        sub.add(VoltageOffset(to_dac_fraction(idle_flux, rail), 0, port=flux_port))
                        if buffer_s:
                            sub.add(IdlePulse(buffer_s))
                        sub.add(Rxy(theta=90.0, phi=-360.0 * detuning * tau, qubit=qubit_name))
                        sub.add(
                            Measure(
                                qubit_name,
                                coords={f"flux_{qubit_name}": flux, f"tau_{qubit_name}": tau},
                                acq_channel=f"S_21_{qubit_name}",
                                **acq,
                            )
                        )
                        sub.add(IdlePulse(4e-9))
            # SAFETY: the flux line back to 0 V at the end of the subschedule
            sub.add(VoltageOffset(0.0, 0, port=flux_port))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
