"""Unit tests for the duration/window knobs on the Qblox READOUT CHANNEL view.

Pure stub elements — no qblox_scheduler needed. The portable contract under
test: values are positive multiples of 4 ns (QM's weights grid — REFUSED
otherwise, no silent rounding), the window never exceeds the pulse (QM cannot
integrate past it), and a pulse shrink clamps the window down with it (the scqo
layer records that echo as a COUPLED change via _sync_coupled).

Greenfield: the view is constructed ENTITY-NAME first (the channel address
``fl.q0`` since SCQO 4.0.0, what scqo addresses and what every refusal must
cite) over the vendor element (``q0``) — one string before the model split them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scqo_qblox.backend.qblox_backend import QbloxReadoutChannel


def _element(pulse_duration=2e-6, integration_time=2e-6):
    return SimpleNamespace(
        name="q0",
        measure=SimpleNamespace(
            pulse_duration=pulse_duration, integration_time=integration_time
        ),
    )


class _Param:
    """Legacy QCoDeS-style parameter: a callable read/write knob."""

    def __init__(self, value):
        self._value = value

    def __call__(self, *args):
        if args:
            self._value = float(args[0])
            return None
        return self._value


def test_duration_and_window_roundtrip():
    view = QbloxReadoutChannel("fl.q0", _element())
    assert view.readout_duration_s == pytest.approx(2.0e-6)
    assert view.readout_integration_s == pytest.approx(2.0e-6)
    view.readout_duration_s = 4.0e-6
    view.readout_integration_s = 3.0e-6
    assert view.readout_duration_s == pytest.approx(4.0e-6)
    assert view.readout_integration_s == pytest.approx(3.0e-6)


def test_duration_shrink_clamps_window():
    el = _element(pulse_duration=2e-6, integration_time=2e-6)
    QbloxReadoutChannel("fl.q0", el).readout_duration_s = 1.0e-6
    assert el.measure.pulse_duration == pytest.approx(1.0e-6)
    assert el.measure.integration_time == pytest.approx(1.0e-6)  # clamped down


def test_duration_grow_leaves_window():
    el = _element(pulse_duration=2e-6, integration_time=2e-6)
    QbloxReadoutChannel("fl.q0", el).readout_duration_s = 4.0e-6
    assert el.measure.pulse_duration == pytest.approx(4.0e-6)
    assert el.measure.integration_time == pytest.approx(2.0e-6)  # independent knob


def test_window_beyond_pulse_refused():
    el = _element(pulse_duration=2e-6, integration_time=2e-6)
    view = QbloxReadoutChannel("fl.q0", el)
    with pytest.raises(ValueError, match="window <= duration") as err:
        view.readout_integration_s = 3.0e-6
    # the refusal cites the ENTITY the user addressed, not the vendor element
    assert str(err.value).startswith("fl.q0:")
    assert el.measure.integration_time == pytest.approx(2.0e-6)  # untouched


def test_off_grid_values_refused():
    view = QbloxReadoutChannel("fl.q0", _element())
    for bad in (1.002e-6, 2.0001e-6, -2.0e-6, 0.0, 3e-9):  # off the 4 ns grid
        with pytest.raises(ValueError, match="multiple of 4 ns"):
            view.readout_duration_s = bad
        with pytest.raises(ValueError, match="multiple of 4 ns"):
            view.readout_integration_s = bad
    assert view.readout_duration_s == pytest.approx(2.0e-6)  # untouched


def test_legacy_callable_parameters_supported():
    """The old QCoDeS API generation (callable params) drives the same logic."""
    el = SimpleNamespace(
        name="q0",
        measure=SimpleNamespace(
            pulse_duration=_Param(2e-6), integration_time=_Param(2e-6)
        ),
    )
    view = QbloxReadoutChannel("fl.q0", el)
    view.readout_duration_s = 1.0e-6
    assert el.measure.pulse_duration() == pytest.approx(1.0e-6)
    assert el.measure.integration_time() == pytest.approx(1.0e-6)  # clamped
    with pytest.raises(ValueError, match="window <= duration"):
        view.readout_integration_s = 2.0e-6
