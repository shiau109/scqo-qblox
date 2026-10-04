"""What only this backend consumes, on top of an experiment's neutral requirements.

The core declares what every backend needs (``scqo/requirements.py``,
``scqo/experiments/_requires.py``). A knob only this driver's probes read is
declared here and added by the driver's subclass:

    requires = (*QubitRamseyPhasor.requires, HALF_PI_AMPLITUDE)

``scqo run <name> --help`` then lists it with the rest.
"""

from __future__ import annotations

from scqo.requirements import Requirement

#: a pi/2 pulse here is the pi pulse's gate at theta=90 (``X90`` / ``Rxy``), so
#: it is played at half of ``pi_amp``; ``pi_amp_x90`` is not realized
HALF_PI_AMPLITUDE = Requirement(
    "pi_amp", "the pi/2 pulses are played at half the pi amplitude")
