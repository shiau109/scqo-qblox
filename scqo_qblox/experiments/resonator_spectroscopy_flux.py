"""Qblox resonator spectroscopy vs flux — supplies only ``probe()``.

cal04 reference pattern: per flux point set a ``VoltageOffset`` on the qubit's own
flux line and let it settle, then sweep the readout frequency across the detuning
window around the current ``readout_freq_hz`` via ``Measure(freq=...)``. Unlike
qubit_spectroscopy_flux_pulse there is no return-to-idle before readout — the resonator
is measured AT the biased flux, the flux-dependent dip IS the signal.

ABSOLUTE frame (no ``_pulse`` in the name): the swept value IS the line voltage —
``VoltageOffset`` REPLACES the standing bias rather than riding on it, so nothing
is added to it here. That is the whole difference from
``qubit_spectroscopy_flux_pulse``, and it is why the two probes must not share an
emission path.

Flux safety: every subschedule ends with the flux line parked at the calibrated
idle when there is one, else at 0 V. The park is a COURTESY, not part of the
measurement, so an uncalibrated line falls back to 0 V instead of refusing the
run — unlike the relative frame, where the same number is load-bearing.
Parameters, the dispersive-model fit and the sweet-spot/f_r0/g reporting are
inherited from ``scqo.experiments.ResonatorSpectroscopyFlux``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ResonatorSpectroscopyFlux

from ._flux_limits import check_flux_bias_absolute, to_dac_fraction
from ._vendor import vendor_element


@register
class QbloxResonatorSpectroscopyFlux(ResonatorSpectroscopyFlux):
    """Build a multiplexed resonator-flux-map Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        if self.params.flux_component is not None:
            raise NotImplementedError(
                "flux_component is not realized on the Qblox backend yet: this "
                "probe sweeps each target's OWN flux line only (an assigned "
                "source would be silently wrong, so it refuses)")
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, VoltageOffset
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        flux_v = self.sweep_axes["flux_bias_v"]
        detuning = self.sweep_axes["detuning_hz"]
        reps = self.params.num_averages

        schedule = Schedule("resonator_spectroscopy_flux_multiplexed")
        for qubit_name in self.params.targets:
            # the flux port is vendor-only; reaching it through the FLUX channel
            # means a target with no flux wiring refuses here, not mid-schedule
            element = vendor_element(self, qubit_name, "flux")
            flux_port = element.ports.flux
            # detuning is relative to the CURRENT readout_freq_hz (readout channel)
            center = self.device.channel(qubit_name, "readout").readout_freq_hz
            # Park target only. The neutral knob refuses when uncalibrated (NaN),
            # which is right for the relative frame but not here: this probe never
            # measures FROM the idle bias, so a missing calibration must not block
            # a run that is otherwise perfectly well defined.
            try:
                idle_flux = self.device.flux_line(qubit_name).idle_flux
            except (ValueError, NotImplementedError, KeyError):
                idle_flux = 0.0
            # ABSOLUTE frame: the swept values ARE the line voltage, so nothing is
            # added to them. Refuse past the module's rail BEFORE compiling — the
            # compiler's own failure is an internal numpy TypeError naming no port.
            rail = check_flux_bias_absolute(
                self, name=f"{qubit_name} flux", port=flux_port, bias_v=flux_v)
            sub = Schedule(f"res_spec_flux_{qubit_name}")
            # flux OUTER, detuning INNER: flat bin order then matches the canonical
            # sweep-axes order (flux_bias_v, detuning_hz). The reps loop sits between
            # them (cal04 pattern) so the flux offset is applied and settled ONCE per
            # flux point; identical coords still average on the cluster.
            #
            # VOLTS -> DAC FRACTION. Every sequencer operand is a fraction of full
            # scale, so the domain is converted here; passing volts through raw
            # emitted value*rail (2.5x on a QCM). The reported axis stays volts —
            # `_to_canonical` rebuilds it from `experiment.sweep_axes`.
            with sub.loop(
                linspace(to_dac_fraction(flux_v[0], rail), to_dac_fraction(flux_v[-1], rail),
                         flux_v.size, dtype=DType.AMPLITUDE)
            ) as flux:
                # 1. bias the qubit's own flux line for this point and let it settle
                sub.add(VoltageOffset(flux, 0, port=flux_port))
                sub.add(IdlePulse(1e-6))  # flux settling (cal04)
                with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                    # 2. sweep the readout frequency across the detuning window
                    with sub.loop(
                        linspace(
                            center + float(detuning[0]),
                            center + float(detuning[-1]),
                            detuning.size,
                            dtype=DType.FREQUENCY,
                        )
                    ) as freq:
                        sub.add(
                            Measure(
                                qubit_name,
                                freq=freq,
                                coords={
                                    f"flux_{qubit_name}": flux,
                                    f"frequency_{qubit_name}": freq,
                                },
                                acq_channel=f"S_21_{qubit_name}",
                            )
                        )
                        sub.add(IdlePulse(10e-6))  # resonator decay (cal04)
            # SAFETY: flux line back to its idle value at the end of the subschedule
            sub.add(VoltageOffset(to_dac_fraction(idle_flux, rail), 0, port=flux_port))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
