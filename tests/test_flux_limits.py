"""The Qblox flux range guard and the volts -> DAC-fraction conversion.

Two things nothing else in this repo checked, both silent on hardware:

1. **Units.** Every sequencer operand is a fraction of full scale, not volts —
   ``VoltageOffset``'s operand is documented "the unitless amplitude", and the
   scheduler's ``max_awg_output_voltage`` is dead metadata. A probe that passed
   ``flux_bias_v`` through unconverted emitted ``value * rail``: a requested
   0.3 V left a QCM as 0.75 V, and every fitted ``flux_offset`` was wrong by the
   module's full-scale factor.
2. **Range.** Measured 2026-07-30: a +/-0.9 domain compiles clean and +/-3.0 dies
   with an internal numpy ``ufunc 'absolute' ... StrDType`` naming no port. There
   is no usable backstop, so the guard must run before compilation.

The rail is fixed by the MODULE here — Qblox has no direct/amplified switch.
"""

import json
from pathlib import Path

import pytest

from scqo_qblox.experiments._flux_limits import (
    check_flux_bias_absolute,
    check_flux_pulse_relative,
    flux_rail_v,
    to_dac_fraction,
)

from conftest import ROSTER_TOML, make_backend, make_experiment

#: the 2q fixture is the one that wires FLUX (module6, a baseband QCM); the
#: minimal fixture has no real_output edges at all.
HW_2Q = Path(__file__).resolve().parent / "fixtures" / "hw_config_2q.json"


class _Exp:
    """Minimal stand-in: the guard only ever reaches backend._hw_agent."""

    def __init__(self, backend):
        self.backend = backend


@pytest.fixture
def exp(tmp_path):
    from scqo.roster import parse_components

    roster = parse_components(ROSTER_TOML)
    hw = json.loads(HW_2Q.read_text(encoding="utf-8"))
    return _Exp(make_backend(tmp_path, roster, hw_config=hw))


# ------------------------------------------------------------------ the rail

def test_a_flux_port_resolves_to_its_module_rail(exp):
    """q1:fl is wired to module6, a QCM -> 5 Vpp = +/-2.5 V peak into 50 ohm.
    The guard compares a single-ended voltage, so it takes HALF the spec: reading
    5 here would be a silent factor of two in the permissive direction."""
    assert flux_rail_v(exp, "q1:fl", name="q1") == 2.5


def test_an_unwired_flux_port_refuses_rather_than_assuming(exp):
    with pytest.raises(ValueError, match="not wired to any baseband"):
        flux_rail_v(exp, "q9:fl", name="q9")


def test_flux_on_an_RF_module_is_refused_as_a_WIRING_error():
    """Three of the four module types in this lab cannot emit DC at all, so this
    is the realistic misconfiguration — and no voltage limit expresses it.

    Driven off a hand-built config rather than a real backend on purpose: the
    VENDOR validator already rejects ``real_output_N`` on an RF module
    ("Invalid channel name specified for module of type QCM_RF"), so this branch
    is unreachable through a config that loads. It stays as the answer for an
    unvalidated dict or a module type we do not know, where the alternative is
    silently assuming a rail."""
    class _Raw:
        pass

    raw = _Raw()
    raw.backend = _Raw()
    raw.backend._hw_agent = _Raw()
    raw.backend._hw_agent.hardware_configuration = {
        "hardware_description": {
            "cluster_A": {"modules": {"10": {"instrument_type": "QCM_RF"}}},
        },
        "connectivity": {"graph": [["cluster_A.module10.real_output_0", "q1:fl"]]},
    }
    with pytest.raises(ValueError, match="cannot emit DC at all"):
        flux_rail_v(raw, "q1:fl", name="q1")


# ------------------------------------------------------------ the conversion

def test_volts_become_a_fraction_of_full_scale():
    """THE bug this module exists for. 0.3 V on a 2.5 V rail is 0.12 of full
    scale — passing 0.3 straight through would emit 0.75 V."""
    assert to_dac_fraction(0.3, 2.5) == pytest.approx(0.12)
    assert to_dac_fraction(-2.5, 2.5) == pytest.approx(-1.0)
    assert to_dac_fraction(0.0, 2.5) == 0.0


def test_the_conversion_is_separate_from_the_checks():
    """A function that validates AND silently rescales is two jobs; the rescale
    has to be visible at the call site or the next probe forgets it."""
    assert to_dac_fraction(1.0, 2.5) == pytest.approx(0.4)  # no validation here


# ------------------------------------------------------------- the two frames

def test_the_absolute_frame_does_not_add_the_idle_bias(exp):
    """set_dc_offset REPLACES the standing bias. A +/-2.0 V sweep is legal even on
    a line parked at 1.0 V, and the relative check on the same numbers is not."""
    assert check_flux_bias_absolute(exp, name="q1", port="q1:fl",
                                    bias_v=[-2.0, 2.0]) == 2.5
    with pytest.raises(ValueError, match="past the port"):
        check_flux_pulse_relative(exp, name="q1", port="q1:fl",
                                  idle_v=1.0, amps_v=[-2.0, 2.0])


def test_the_relative_frame_checks_the_SUM(exp):
    """The DAC adds offsets and waveforms, so a window that is fine alone still
    clips once it rides on a standing bias."""
    check_flux_pulse_relative(exp, name="q1", port="q1:fl",
                              idle_v=0.0, amps_v=[-2.0, 2.0])
    with pytest.raises(ValueError, match="past the port") as err:
        check_flux_pulse_relative(exp, name="q1", port="q1:fl",
                                  idle_v=1.0, amps_v=[-2.0, 2.0])
    assert "SILENTLY" in str(err.value)  # why it refuses instead of warning


def test_the_relative_check_is_asymmetric(exp):
    """A positive idle eats headroom on the positive side only."""
    with pytest.raises(ValueError, match="past the port"):
        check_flux_pulse_relative(exp, name="q1", port="q1:fl",
                                  idle_v=1.0, amps_v=[-1.6, 1.6])
    check_flux_pulse_relative(exp, name="q1", port="q1:fl",
                              idle_v=1.0, amps_v=[-1.6, 0.0])


# --------------------------------------------------------------- the warning

def test_near_full_scale_warns_about_the_termination_assumption(exp):
    """The 2.5 V rail assumes 50 ohm. Unterminated, the swing roughly doubles —
    onto the module's absolute maximum rating."""
    with pytest.warns(RuntimeWarning, match="50 ohm"):
        check_flux_bias_absolute(exp, name="q1", port="q1:fl", bias_v=[2.4])


def test_a_normal_window_is_silent(exp):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_flux_bias_absolute(exp, name="q1", port="q1:fl",
                                        bias_v=[-0.3, 0.3]) == 2.5


# ------------------------------------------------------- end to end on a probe

def test_the_resonator_flux_probe_emits_FRACTIONS_not_volts(tmp_path):
    """The regression that matters: a 0.3 V request must reach the schedule as
    0.12 of full scale, not as 0.3."""
    from scqo.roster import parse_components

    from scqo_qblox.experiments.resonator_spectroscopy_flux import (
        QbloxResonatorSpectroscopyFlux as C,
    )

    roster = parse_components(ROSTER_TOML)
    hw = json.loads(HW_2Q.read_text(encoding="utf-8"))
    backend = make_backend(tmp_path, roster, hw_config=hw)
    params = C.Parameters(targets=["q1"], start_flux_v=-0.3, end_flux_v=0.3,
                          num_flux_points=5, num_averages=2)
    experiment = make_experiment(C, backend, roster, params)
    experiment.sweep_axes = experiment.define_sweep()
    schedule = experiment.probe()

    domains = _flux_loop_domains(schedule)
    assert domains, "probe emitted no swept amplitude domain"
    start, stop = domains[0]
    # +/-0.3 V on a 2.5 V rail -> +/-0.12 of full scale. Unconverted it would be
    # +/-0.3 here, and the DAC would emit +/-0.75 V.
    assert (start, stop) == pytest.approx((-0.12, 0.12), abs=1e-9)
    # the REPORTED axis stays volts -- only the emission is converted
    assert experiment.sweep_axes["flux_bias_v"].max() == pytest.approx(0.3)


def test_the_park_offset_is_converted_too(tmp_path):
    """The end-of-subschedule park is a real number in the schedule, so it is the
    one emission whose conversion can be read off directly."""
    from scqo.roster import parse_components

    from scqo_qblox.experiments.resonator_spectroscopy_flux import (
        QbloxResonatorSpectroscopyFlux as C,
    )

    roster = parse_components(ROSTER_TOML)
    hw = json.loads(HW_2Q.read_text(encoding="utf-8"))
    backend = make_backend(tmp_path, roster, hw_config=hw)
    params = C.Parameters(targets=["q1"], start_flux_v=-0.3, end_flux_v=0.3,
                          num_flux_points=5, num_averages=2)
    experiment = make_experiment(C, backend, roster, params)
    experiment.sweep_axes = experiment.define_sweep()
    schedule = experiment.probe()

    parks = _static_flux_offsets(schedule)
    assert parks, "probe emitted no static flux offset"
    # whatever the fixture's sweet spot is, the emitted fraction must be it/2.5
    for level in parks:
        assert abs(level) <= 1.0
        assert abs(level * 2.5) <= 2.5


def _walk(node, seen=None):
    """Every operation in a schedule tree.

    ``LoopOperation.body`` is a separate child from ``operations`` — a walker
    that follows only ``operations`` silently finds nothing inside any loop,
    which is where the swept flux lives.
    """
    seen = seen if seen is not None else set()
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    children = list((getattr(node, "operations", None) or {}).values())
    body = getattr(node, "body", None)
    if body is not None:
        children.append(body)
    for child in children:
        yield from _walk(child, seen)


def _static_flux_offsets(schedule) -> list[float]:
    """Flux-port VoltageOffset levels that are plain numbers (not loop vars)."""
    found = []
    for node in _walk(schedule):
        info = (getattr(node, "data", None) or {}).get("pulse_info")
        if not isinstance(info, dict) or "offset_path_I" not in info:
            continue
        if "fl" not in str(info.get("port", "")):
            continue
        value = info["offset_path_I"]
        if isinstance(value, (int, float)):
            found.append(float(value))
    return found


def _flux_loop_domains(schedule) -> list[tuple[float, float]]:
    """(start, stop) of every AMPLITUDE loop domain in the tree.

    ``LoopOperation.domain`` is a ``{loop_var: LinearDomain}`` mapping, not a bare
    domain — the swept flux values live here, not on the ``VoltageOffset`` (whose
    operand is just the loop variable).
    """
    found = []
    for node in _walk(schedule):
        domain = getattr(node, "domain", None)
        if not isinstance(domain, dict):
            continue
        for linear in domain.values():
            # DType is a str-Enum: str() yields 'amplitude', not 'DType.AMPLITUDE'
            if not str(getattr(linear, "dtype", "")).upper().endswith("AMPLITUDE"):
                continue
            start, stop = getattr(linear, "start", None), getattr(linear, "stop", None)
            if isinstance(start, (int, float)) and isinstance(stop, (int, float)):
                found.append((float(start), float(stop)))
    return found
