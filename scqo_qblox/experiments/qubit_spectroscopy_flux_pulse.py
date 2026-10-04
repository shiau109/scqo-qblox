"""Qblox pulsed flux qubit spectroscopy — supplies only ``probe()``.

cal07b-style CW pattern: per flux point set a ``VoltageOffset`` on the qubit's own
flux line, step the drive clock across the detuning window around the current
``drive_freq_hz``, apply a weak CW SATURATION drive (``view.drive_amp`` — the
``drive_power_dbm`` residual parked on ``spec.spec_amp``) held through the driven
dwell, then return the flux to its idle value BEFORE measuring (clean readout at
the calibrated operating point, matching the QM flux probe). That return-to-idle
IS the ``_pulse`` name's probe contract: every slice reads out at the same
idle-flux condition, so the neutral ``estimate()`` reduces the whole map against
ONE global IQ reference. Unlike the old cal07
X-pulse version this needs no calibrated pi, so it works during bring-up. Flux
safety: every subschedule ends with the drive off and the flux line back at 0 V.

Parameters, the transmon-arch fit and the sweet-spot/Ej_sum reporting are
inherited from ``scqo.experiments.QubitSpectroscopyFluxPulse``. The core ``run()``
parks ``drive_power_dbm`` (a recorded set -> revert) before this probe reads
``view.drive_amp``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitSpectroscopyFluxPulse

from scqo.experiments._capabilities import flux_anchor_v

from ._flux_limits import check_flux_pulse_relative, to_dac_fraction
from ._reset import add_reset
from ._vendor import vendor_element


@register
class QbloxQubitSpectroscopyFluxPulse(QubitSpectroscopyFluxPulse):
    """Build a multiplexed pulsed flux-spectroscopy Schedule for a Qblox cluster."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_spectroscopy_flux_pulse --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the drive is a continuous tone held for the thermal-reset wait, not a pulse after "
        "it: the wait is the saturation time",
        "the flux is a voltage offset held together with the drive and returned to the idle "
        "flux 65 ns before the readout",
        "flux_component is refused: only the target's own flux line can be swept",
    )

    def probe(self) -> Any:
        if self.params.flux_component is not None:
            raise NotImplementedError(
                "flux_component is not realized on the Qblox backend yet: this "
                "probe sweeps each target's OWN flux line only (an assigned "
                "source would be silently wrong, so it refuses)")
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            VoltageOffset,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        flux_v = self.sweep_axes["flux_bias_v"]
        detuning = self.sweep_axes["detuning_hz"]
        reps = self.params.num_averages

        schedule = Schedule("qubit_spectroscopy_flux_pulse_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "drive")
            # ports are vendor-only; each is reached through the channel whose
            # FUNCTION it carries, so a target with no flux wiring refuses here
            element = vendor_element(self, qubit_name, "flux")
            flux_port = element.ports.flux
            microwave_port = vendor_element(self, qubit_name, "drive").ports.microwave
            center = view.drive_freq_hz  # detuning is relative to the CURRENT drive_freq_hz
            drive_amp = float(view.drive_amp)  # run() parked the solved spec_amp residual
            drive_clock = f"{qubit_name}.01"
            # The RELATIVE frame's origin. Read through the capability's own
            # anchor, the same call `estimate()` uses to record `old_idle_flux`,
            # so the bias this probe EMITS from and the one the fit
            # re-references against cannot drift apart. Uncalibrated refuses
            # here rather than defaulting to 0 -- at zero idle the relative and
            # absolute frames coincide, which is exactly what hid this probe
            # emitting an absolute sweep under a `_pulse` name until 2026-07-30.
            idle_flux = flux_anchor_v(self, qubit_name)
            # The DAC emits the SUM, so the check is idle + excursion — a window
            # that is fine alone can still clip once it rides on the standing bias.
            rail = check_flux_pulse_relative(
                self, name=f"{qubit_name} flux", port=flux_port,
                idle_v=idle_flux, amps_v=flux_v)
            sub = Schedule(f"qubit_spec_flux_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                # flux OUTER, detuning INNER: flat bin order then matches the
                # canonical sweep-axes order (flux_bias_v, detuning_hz)
                # RELATIVE frame: the emitted level is idle + excursion, so the
                # DOMAIN carries the offset and the loop variable is already the
                # absolute line voltage — then converted to the DAC fraction the
                # sequencer actually consumes. Shifting the domain rather than
                # writing `idle_flux + flux` at the use site is not a style
                # choice: an arithmetic expression on a loop variable reaches the
                # compiler as a BinaryExpression and dies in
                # `expand_awg_from_normalised_range` with `ufunc 'absolute' did
                # not contain a loop ... StrDType`.
                #
                # The reported axis stays the RELATIVE window in VOLTS:
                # `_to_canonical` rebuilds coordinates from
                # `experiment.sweep_axes`, and the coord below only has to be
                # distinct per point for the cluster's binning. `old_idle_flux`
                # records the origin; the fit re-references
                # `absolute = old_idle_flux + fitted`.
                with sub.loop(
                    linspace(to_dac_fraction(idle_flux + float(flux_v[0]), rail),
                             to_dac_fraction(idle_flux + float(flux_v[-1]), rail),
                             flux_v.size, dtype=DType.AMPLITUDE)
                ) as flux:
                    with sub.loop(
                        linspace(
                            center + float(detuning[0]),
                            center + float(detuning[-1]),
                            detuning.size,
                            dtype=DType.FREQUENCY,
                        )
                    ) as freq:
                        # 1. bias the qubit's own flux line for this point. `flux`
                        #    is already idle + excursion (see the domain above);
                        #    VoltageOffset is sticky, so holding it here and
                        #    returning to `idle_flux` at step 3 emits exactly what
                        #    QM's initialize_qpu + play("const") does.
                        sub.add(VoltageOffset(flux, 0, port=flux_port))
                        sub.add(IdlePulse(4e-9))
                        # 2. weak CW saturation drive at the shifted frequency, held
                        #    through the driven dwell (Reset = the steady-state wait,
                        #    the cal05/cal07b idiom) — no calibrated pi needed
                        sub.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                        sub.add(VoltageOffset(drive_amp, 0, port=microwave_port, clock=drive_clock))
                        # this experiment's Parameters carry no reset_method at all
                        # (it is flux-tagged, not qubit_reset-tagged), so add_reset
                        # always resolves thermal here; it goes through the one door
                        # so that "no probe builds the gate itself" stays checkable.
                        add_reset(sub, self, qubit_name)
                        sub.add(VoltageOffset(0, 0, port=microwave_port, clock=drive_clock))
                        # 3. flux back to idle BEFORE the readout (measure at the
                        #    calibrated operating point, matching the QM flux probe).
                        #    Excursion 0 == parked here, which is what makes this
                        #    the relative frame's origin rather than the DAC zero.
                        sub.add(VoltageOffset(to_dac_fraction(idle_flux, rail), 0, port=flux_port))
                        sub.add(IdlePulse(4e-9))
                        sub.add(IdlePulse(61e-9))  # QRC-with-QCM settling fudge (cal07)
                        sub.add(
                            Measure(
                                qubit_name,
                                coords={
                                    f"flux_{qubit_name}": flux,
                                    f"frequency_{qubit_name}": freq,
                                },
                                acq_channel=f"S_21_{qubit_name}",
                            )
                        )
            # SAFETY: drive off + flux line back to 0 V at the end of the subschedule
            sub.add(VoltageOffset(0, 0, port=microwave_port, clock=drive_clock))
            sub.add(VoltageOffset(0.0, 0, port=flux_port))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
