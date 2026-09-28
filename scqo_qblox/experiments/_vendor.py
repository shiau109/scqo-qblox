"""Raw-vendor access for the probes — the one place that leaves the neutral surface.

Since the greenfield model a probe reads its NEUTRAL state through the channel
entity that owns the knob (``self.device.channel(target, "readout")``), which
serves the SCQO runtime value whether ``self.device`` is the Session's
RecordingDevice or a bare ``QbloxDeviceModel``. A few things a schedule needs
are vendor-only and have no neutral field at all — the element's ports
(``ports.flux``, ``ports.microwave``) and the vendor's flux sweet-spot slot — so
they come from the qblox ``DeviceElement`` itself, addressed through the roster
here.
"""

from __future__ import annotations

import math
from typing import Any


def vendor_element(experiment: Any, target: str, kind: str) -> Any:
    """The raw qblox ``DeviceElement`` behind ``target``'s DEFAULT channel of
    one kind.

    Entity names are resolved through the ROSTER, never by string arithmetic:
    ask for the default channel name, then address the vendor tree by that name
    (the core power_chain idiom — ``DeviceModel`` has no ``channel()``). Going
    through the channel KIND is also the capability guard: a target with no
    channel of that kind fails on a clear roster error here, instead of on a
    missing vendor port deep inside a schedule.
    """
    name = experiment.device.roster.default_channel(target, kind)
    return experiment.backend.device.component(name)._element


# `idle_flux` used to live here, reading element.flux_params.sweet_spot directly
# and falling back to 0.0. It is now the REALIZED neutral knob on the flux LINE
# view (backend/fieldmap.py + QbloxFluxLine), so a probe reads it the same way it
# reads every other governed value - through the anchor the fit re-references:
#
#     flux_anchor_v(self, target)       # = self.device.flux_line(target).idle_flux
#
# Do not reintroduce a vendor shortcut. The 0.0 fallback is what made the wrong
# frame invisible: at zero idle the absolute and relative sweeps coincide exactly,
# so an uncalibrated line silently produced a correct-looking absolute sweep from
# a probe whose contract says relative. The knob now refuses on NaN instead.
