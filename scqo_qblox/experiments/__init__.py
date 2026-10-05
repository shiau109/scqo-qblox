"""Importing this package registers every Qblox experiment into the scqo catalog.

Add a line here for each new experiment module so its ``@register`` runs.
"""

from . import qubit_deterministic_benchmarking  # noqa: F401  (import side effect: @register)
from . import qubit_drag_equator  # noqa: F401  (import side effect: @register)
from . import qubit_echo  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_continuous  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_discrete  # noqa: F401  (import side effect: @register)
from . import qubit_power_rabi  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey_cryoscope  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey_phasor  # noqa: F401  (import side effect: @register)
from . import qubit_relaxation  # noqa: F401  (import side effect: @register)
from . import qubit_resonator_stark  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_cryoscope  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_sqrb  # noqa: F401  (import side effect: @register)
from . import qubit_thermal_population  # noqa: F401  (import side effect: @register)
from . import qubit_tomography  # noqa: F401  (import side effect: @register)
from . import readout_frequency  # noqa: F401  (import side effect: @register)
from . import readout_power  # noqa: F401  (import side effect: @register)
from . import readout_time_of_flight  # noqa: F401  (import side effect: @register)
from . import broadband_qubit_spectroscopy  # noqa: F401  (import side effect: @register)
from . import broadband_resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_flux  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_chain  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_amp  # noqa: F401  (import side effect: @register)
from . import single_shot_readout  # noqa: F401  (import side effect: @register)

# Active reset is opt-in per probe here (_reset.py, default DENY). scqo lists what a
# reset needs from the shared Parameters, which accept reset_method="active" on every
# backend; this tells it where the opt-in is declared, so `scqo run <name> --help`
# leaves those lines out for a probe that refuses the setting.
from scqo.requirements import declare_probe_opt_in as _declare_probe_opt_in

from ._reset import ACTIVE_RESET_ATTR as _ACTIVE_RESET_ATTR

_declare_probe_opt_in("scqo_qblox", "reset_method", "active", _ACTIVE_RESET_ATTR)

__all__ = [
    "broadband_qubit_spectroscopy",
    "broadband_resonator_spectroscopy",
    "qubit_deterministic_benchmarking",
    "qubit_drag_equator",
    "resonator_spectroscopy",
    "qubit_spectroscopy",
    "qubit_spectroscopy_flux_pulse",
    "qubit_sqrb",
    "qubit_ramsey",
    "qubit_ramsey_cryoscope",
    "qubit_ramsey_flux_pulse",
    "qubit_ramsey_phasor",
    "qubit_spectroscopy_cryoscope",
    "qubit_power_rabi",
    "resonator_spectroscopy_flux",
    "resonator_spectroscopy_power_amp",
    "resonator_spectroscopy_power_chain",
    "readout_power",
    "readout_time_of_flight",
    "readout_frequency",
    "qubit_relaxation",
    "qubit_resonator_stark",
    "qubit_echo",
    "qubit_thermal_population",
    "qubit_tomography",
    "qubit_parity_switch_continuous",
    "qubit_parity_switch_discrete",
    "single_shot_readout",
]
