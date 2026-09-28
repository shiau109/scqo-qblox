"""The output attenuator's ceiling is a property of the MODULE, not of this driver.

chipA, 2026-07-29: ``scqo run qubit_spectroscopy_overlap`` (that experiment has
since been merged into ``qubit_spectroscopy`` as ``readout_overlap=true``) died with
``38 is invalid: must be between 0 and 30 inclusive; Parameter:
cluster_A_module4.out0_att``. The drive-power solve had picked 38 dB against a
hardcoded 0..60 range, and slot 4 (a QCM-RF, ISA 2.1) attenuates at most 30 —
while slot 8 on the same cluster (a QRM-RF, ISA 2.0) runs at 40 today. So the
ceiling is neither a constant nor a function of the module type:
qblox_instruments builds the ``out<k>_att`` validator from a LIVE SCPI query, and
nothing connects until a run.

The design that follows from that, and what this file pins:
  * the solve takes the ceiling as an argument and clamps to it, which costs DAC
    range and never changes the power at the port;
  * the ceiling is asked for only when a chain was solved above an attenuation
    every Qblox RF output supports — otherwise ``acquire()`` must not so much as
    dial the cluster;
  * what is learned is remembered in a sidecar, so the NEXT process solves right
    the first time.
"""

from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import make_backend  # noqa: E402

from scqo_qblox.backend.qblox_backend import (  # noqa: E402
    ATT_LIMITS_FILE,
    QBLOX_NOMINAL_FULL_SCALE_DBM,
    _att_limits,
    _module_max_att,
    _port_outputs,
    _solve_att,
)


@pytest.fixture()
def backend(tmp_path, roster):
    return make_backend(tmp_path, roster)


# ------------------------------------------------------------------- the solve

def test_the_ceiling_clamps_the_attenuation_not_the_power():
    """THE regression. -40 dBm wants 38 dB of attenuation; on a 30 dB output it
    must come out at 30 with the amplitude absorbing the missing 8 dB — the same
    power at the port, just less DAC range. Silently pushing 38 is what the
    instrument refused."""
    target = -40.0
    wide_att, wide_amp = _solve_att("xy1.q1", target, "spec_amp", 60)
    narrow_att, narrow_amp = _solve_att("xy1.q1", target, "spec_amp", 30)

    assert wide_att == 38 and narrow_att == 30
    assert narrow_amp < wide_amp  # the cost is dynamic range...
    for att, amp in ((wide_att, wide_amp), (narrow_att, narrow_amp)):
        # ...never the delivered power
        assert QBLOX_NOMINAL_FULL_SCALE_DBM - att + 20 * math.log10(amp) == pytest.approx(target)


def test_the_ceiling_never_makes_the_amplitude_illegal():
    """Clamping only ever LOWERS the amplitude, so the canonical <= 0.5 operating
    point cannot be broken by a narrower attenuator (the >0.5 warning belongs to
    a target too HIGH for the chain, which is a different failure)."""
    for max_att in (20, 30, 60):
        _att, amp = _solve_att("xy1.q1", -40.0, "spec_amp", max_att)
        assert 0 < amp <= 0.5


# ------------------------------------------------------- resolving the module

def test_connectivity_resolves_a_port_to_its_physical_output(backend):
    """A ceiling belongs to a (cluster, slot, output), and the connectivity graph
    is the only place the config says which one a port is on."""
    outputs = _port_outputs(backend._hw_agent.hardware_configuration)
    assert "q1:res" in outputs and "q1:mw" in outputs
    for cluster, slot, output in outputs.values():
        assert isinstance(cluster, str) and isinstance(slot, int) and output >= 0


def test_module_max_att_reads_a_real_qblox_module():
    """Against the vendor class, not a stand-in: the fakes below pin the SHAPES,
    but only a real ``qblox_instruments`` module proves the attribute path
    (``module.out<k>_att.vals.max_value``) is the one that exists. The dummy
    transport answers the same ``_get_max_out_att`` query the hardware does."""
    from qblox_instruments import Cluster, ClusterType
    from qcodes.instrument import Instrument

    try:
        Instrument.find_instrument("att_limits_fake").close()
    except KeyError:
        pass
    cluster = Cluster("att_limits_fake",
                      dummy_cfg={4: ClusterType.CLUSTER_QCM_RF,
                                 8: ClusterType.CLUSTER_QRM_RF})
    try:
        for slot in (4, 8):
            ceiling = _module_max_att(getattr(cluster, f"module{slot}"), 0)
            assert isinstance(ceiling, int) and ceiling > 0 and ceiling % 2 == 0
    finally:
        cluster.close()


def test_module_max_att_reads_both_shapes():
    """QCM/QRM expose the ceiling on the parameter's validator (built from the
    live query); QRC has no validator ceiling and a getter instead, because its
    limit moves with the centre frequency. Floored to even dB — the solve emits
    even dB. Neither present -> None, i.e. 'unknowable', never a guess."""
    class Validator:
        max_value = 30

    class QcmLike:
        out0_att = type("P", (), {"vals": Validator()})()

    class QrcLike:
        def get_max_out0_att(self):
            return 31.5

    assert _module_max_att(QcmLike(), 0) == 30
    assert _module_max_att(QrcLike(), 0) == 30  # 31.5 floored onto the 2 dB grid
    assert _module_max_att(object(), 0) is None


# ------------------------------------------- when the cluster is asked, and not

def test_a_modest_attenuation_never_dials_the_cluster(backend, monkeypatch):
    """The offline contract. Every chain in the fixture sits below the
    universally-safe attenuation, so there is nothing a ceiling could change and
    ``_sync_att_limits`` must not connect — this is what keeps the stubbed-run
    tests (and any future one) off the network."""
    def explode():
        raise AssertionError("_sync_att_limits dialled the cluster unnecessarily")

    monkeypatch.setattr(backend._hw_agent, "get_clusters", explode)
    assert backend._suspect_chains() == {}
    backend._sync_att_limits()  # must not raise


def test_a_high_attenuation_makes_the_chain_suspect(backend):
    """...and one solved past that point does have to be checked."""
    backend.device.component("xy1.q1").drive_power_dbm = -40.0
    suspect = backend._suspect_chains()
    assert "q1:mw-q1.01" in suspect
    assert suspect["q1:mw-q1.01"].name == "xy1.q1"

    # once the ceiling is known and the value fits under it, the question is
    # settled and the cluster is never asked again
    _att_limits(backend._hw_agent)["q1:mw-q1.01"] = 60
    assert backend._suspect_chains() == {}


def test_an_unreachable_cluster_warns_and_leaves_the_solve_alone(backend, monkeypatch):
    """Cosmetic knowledge must degrade to 'you will hear about it from the
    instrument', never to a failed run before the schedule is even built."""
    backend.device.component("xy1.q1").drive_power_dbm = -40.0

    def unreachable():
        raise TimeoutError("timed out")

    monkeypatch.setattr(backend._hw_agent, "get_clusters", unreachable)
    with pytest.warns(UserWarning, match="could not read the output-attenuation limits"):
        backend._sync_att_limits()
    opts = backend._hw_agent.hardware_configuration.hardware_options
    assert opts.output_att["q1:mw-q1.01"] == 38  # untouched, still the optimistic solve


# ------------------------------------------------- the correction, end to end

class _FakeModule:
    def __init__(self, ceiling):
        self.out0_att = type("P", (), {"vals": type("V", (), {"max_value": ceiling})()})()


class _FakeCluster:
    def __init__(self, **modules):
        for name, mod in modules.items():
            setattr(self, name, mod)


def test_a_narrow_output_is_re_solved_before_the_probe_runs(backend, monkeypatch):
    """The chipA failure, end to end: solve -40 dBm optimistically, discover the
    output only reaches 30 dB, and land on a legal attenuation carrying exactly
    the same power. The re-solve goes through the RAW view, so it is a hardware
    correction and not a calibration proposal."""
    view = backend.device.component("xy1.q1")
    view.drive_power_dbm = -40.0
    opts = backend._hw_agent.hardware_configuration.hardware_options
    assert opts.output_att["q1:mw-q1.01"] == 38  # what the instrument refused

    cluster, slot, _out = _port_outputs(backend._hw_agent.hardware_configuration)["q1:mw"]
    monkeypatch.setattr(
        backend._hw_agent, "get_clusters",
        lambda: {cluster: _FakeCluster(**{f"module{slot}": _FakeModule(30)})},
    )
    with pytest.warns(UserWarning, match="attenuates at most 30 dB"):
        backend._sync_att_limits()

    assert opts.output_att["q1:mw-q1.01"] == 30
    assert view.drive_power_dbm == pytest.approx(-40.0)  # the whole point
    assert 0 < view.drive_amp <= 0.5


def test_the_discovered_ceiling_is_remembered_for_the_next_process(
        backend, tmp_path, roster, monkeypatch):
    """A sidecar beside hw_config.json (mixer_cal.json's neighbour), keyed
    PHYSICALLY by slot/output — the ceiling belongs to the module, so rewiring a
    port must not inherit the old one. A new backend over the same folder solves
    right the first time, with no cluster and no warning."""
    backend.device.component("xy1.q1").drive_power_dbm = -40.0
    cluster, slot, out = _port_outputs(backend._hw_agent.hardware_configuration)["q1:mw"]
    monkeypatch.setattr(
        backend._hw_agent, "get_clusters",
        lambda: {cluster: _FakeCluster(**{f"module{slot}": _FakeModule(30)})},
    )
    with pytest.warns(UserWarning):
        backend._sync_att_limits()

    sidecar = tmp_path / ATT_LIMITS_FILE
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {f"slot{slot}/out{out}": 30}

    fresh = make_backend(tmp_path, roster)
    assert _att_limits(fresh._hw_agent)["q1:mw-q1.01"] == 30
    fresh.device.component("xy1.q1").drive_power_dbm = -40.0
    fresh_opts = fresh._hw_agent.hardware_configuration.hardware_options
    assert fresh_opts.output_att["q1:mw-q1.01"] == 30  # right on the first solve
    assert fresh._suspect_chains() == {}  # ...so nothing to ask the cluster
