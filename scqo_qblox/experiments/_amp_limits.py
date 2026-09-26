"""The DAC ceiling on an amplitude sweep — refused BY NAME, before compiling.

Every qblox_scheduler operand is a fraction of full scale, and the compiler
refuses anything outside ``[-1, 1]`` in
``backends/qblox/qasm_program.py::expand_awg_from_normalised_range``. Three
reasons that error is not good enough on its own:

* it names ``awg_gain`` — an internal parameter the operator never set — instead
  of the knob they did set (``start_amp_factor`` / ``end_amp_factor`` — either
  edge may be the larger, the window being a traversal order) or the stored
  amplitude it multiplies;
* it fires deep inside compilation, after the schedule is built;
* the element-level validator that WOULD have caught it
  (``measure.pulse_amp`` is ``Numbers(0, 1)``) is bypassed entirely by the
  ``device_overrides`` path these probes use — the override is substituted into
  the pulse factory's kwargs and never goes through the parameter setter.

So the probes pre-check here and raise with the neutral knob, the stored
amplitude, and the factor that would fit. This is the amplitude sibling of
``_flux_limits`` on the QM side: same rule, different vendor bound.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = ["MAX_DAC_FRACTION", "check_amp_window"]

#: A qblox sequencer operand is normalized to DAC full scale; the compiler
#: refuses ``|value| > 1``.
MAX_DAC_FRACTION = 1.0


def check_amp_window(prefactors: Iterable[float], base: float, *, target: str,
                     field: str, knob: str = "start_amp_factor/end_amp_factor"
                     ) -> np.ndarray:
    """Validate ``prefactors * base`` against the DAC, returning the absolute amps.

    Args:
        prefactors: the swept factors (scqo's ``amp_prefactor`` axis).
        base: the target's stored amplitude the factors multiply.
        target: qubit name, for the message.
        field: the neutral knob holding ``base`` (``pi_amp`` / ``readout_amp``).
        knob: the neutral parameter to lower.

    Returns:
        The absolute amplitudes, IN THE ORDER GIVEN (a descending window stays
        descending), so callers use the validated values rather than recomputing
        the product.

    Raises:
        ValueError: when any point would exceed the sequencer's normalized range.
    """
    factors = np.asarray(list(prefactors), dtype=float)
    amps = factors * float(base)
    if factors.size == 0:
        return amps
    peak = float(np.max(np.abs(amps)))
    if peak > MAX_DAC_FRACTION:
        worst = float(np.max(np.abs(factors)))
        headroom = MAX_DAC_FRACTION / abs(float(base)) if base else float("inf")
        raise ValueError(
            f"{target}: {knob}={worst:.4g} x {field}={base:.4g} = {peak:.4g} "
            f"exceeds the sequencer's normalized range of {MAX_DAC_FRACTION}. "
            f"Lower {knob} to <= {headroom:.4g}, or lower {field} "
            f"(re-solve the output chain via its *_power_dbm twin)."
        )
    return amps
