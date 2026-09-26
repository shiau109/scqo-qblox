"""The DAC ceiling on an amplitude sweep, refused by name before compiling.

The compiler DOES refuse an over-range amplitude — but deep inside
``expand_awg_from_normalised_range``, naming ``awg_gain``, an internal parameter
the operator never set. And the element validator that would otherwise catch it
(``measure.pulse_amp`` is ``Numbers(0, 1)``) is bypassed entirely by the
``device_overrides`` path these probes use. So the pre-check is the only place
the failure can name the knob the operator actually set.
"""

import numpy as np
import pytest

from scqo_qblox.experiments._amp_limits import MAX_DAC_FRACTION, check_amp_window


def test_a_window_inside_the_dac_returns_the_absolute_amps():
    amps = check_amp_window([0.4, 1.0, 1.8], 0.5, target="q1", field="readout_amp")
    np.testing.assert_allclose(amps, [0.2, 0.5, 0.9])


def test_the_ceiling_is_the_product_not_the_factor():
    """A factor of 1.8 is fine at readout_amp 0.5 and impossible at 0.64 — the
    bound lives on the PRODUCT, which is exactly why a neutral factor cap cannot
    express it and the check has to happen in the driver."""
    check_amp_window([1.8], 0.5, target="q1", field="readout_amp")
    with pytest.raises(ValueError):
        check_amp_window([1.8], 0.64, target="q1", field="readout_amp")


def test_the_message_names_the_knob_the_operator_set():
    """Not `awg_gain`. It must carry the neutral knob, the stored amplitude, and
    the factor that WOULD fit — everything needed to fix it without reading the
    vendor stack."""
    with pytest.raises(ValueError) as err:
        check_amp_window([0.9, 2.0], 0.64, target="q1", field="pi_amp")
    message = str(err.value)
    assert "q1" in message
    assert "start_amp_factor/end_amp_factor" in message
    assert "pi_amp" in message
    assert "awg_gain" not in message
    # the remedy: 1.0 / 0.64 = 1.5625
    assert "1.562" in message


def test_exactly_at_full_scale_is_allowed():
    """The compiler's bound is inclusive (``|value| > 1`` refuses), so a sweep
    landing exactly on full scale is legal and must not be refused."""
    amps = check_amp_window([1.0], MAX_DAC_FRACTION, target="q1", field="pi_amp")
    assert amps[0] == pytest.approx(MAX_DAC_FRACTION)


def test_an_empty_sweep_is_not_an_error():
    assert check_amp_window([], 0.5, target="q1", field="pi_amp").size == 0
