"""The coupler-as-a-target fields are refused BY NAME on this backend.

SCQO's ``drive_line`` and ``mapped_readout`` capabilities (docs/coupler-transmon-plan.md)
drive a target through a named line's channel - usually BORROWED - and read it through
a pair member. v1 realizes them on QM (MW-FEM) only: this backend adopts no borrowed
channel and plays no mapped readout. Two doors refuse, each naming the reason:

* the Session's roster gate asks this device model for the borrowed channel
  (``xy2.c12``) and surfaces its KeyError - pinned here on the model itself;
* ``qubit_power_rabi``'s probe refuses either field before building anything,
  including a ``drive_line`` naming the target's OWN line, which the gate lets
  through (that channel is designed, so the model realizes it).
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import make_backend, make_experiment  # noqa: E402

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402


@pytest.mark.parametrize("fields, named", [
    ({"targets": ["c12"], "drive_line": "xy2", "readout_member": "q1",
      "use_state_discrimination": True}, ["drive_line='xy2'", "readout_member='q1'"]),
    ({"targets": ["q1"], "drive_line": "xy2"}, ["drive_line='xy2'"]),
    ({"targets": ["q1"], "drive_line": "xy1"}, ["drive_line='xy1'"]),
])
def test_power_rabi_refuses_the_coupler_fields_by_name(tmp_path, roster, fields, named):
    cls = get("qubit_power_rabi")
    exp = make_experiment(cls, make_backend(tmp_path, roster), roster,
                          cls.Parameters(num_averages=2, **fields))
    with pytest.raises(NotImplementedError, match="not realized on the Qblox backend") as err:
        exp.probe()
    for text in named:
        assert text in str(err.value)


def test_power_rabi_without_the_fields_still_compiles(tmp_path, roster):
    """The refusal is keyed on the fields alone: the qubits' own run is untouched."""
    from conftest import compile_probe

    cls = get("qubit_power_rabi")
    backend = make_backend(tmp_path, roster)
    # end_amp_factor as test_probe_surface: the fixture's pi_amp x the stock top
    # factor exceeds the DAC range, which is not what this test is about
    exp = make_experiment(cls, backend, roster, cls.Parameters(
        targets=["q1"], num_averages=2, num_amp_points=5, end_amp_factor=0.5))
    assert compile_probe(backend, exp) is not None


def test_the_gate_hears_why_a_borrowed_drive_line_cannot_run(tmp_path, roster):
    """What the Session's drive_line gate surfaces on this backend."""
    backend = make_backend(tmp_path, roster)
    with pytest.raises(KeyError, match="adopts no borrowed channel yet"):
        backend.device.component("xy2.c12")
