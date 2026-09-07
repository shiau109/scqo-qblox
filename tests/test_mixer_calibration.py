"""``scripts/calibrate_mixers.py`` — the AMC operations tool.

Two halves, tested separately:

* the PLAN is pure — hw_config.json + dut_config.json in, ``(slot, output, LO)`` groups
  with per-sequencer NCOs out. No vendor import, no cluster.
* the RUN needs a Cluster, so it uses ``dummy_cfg={slot: ClusterType.CLUSTER_*_RF}``.
  On a dummy the calibration calls themselves are no-ops, which is exactly the point:
  what these tests cover is the CONTROL FLOW around them — the channel map, the LO/NCO
  setup, the tone, the running-sequencer snapshot/restart, and the cache.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("qblox_instruments")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import calibrate_mixers as cm  # noqa: E402


# --------------------------------------------------------------------------- fixtures
def _hw(modules, graph, modulation, output_att=None):
    return {
        "config_type": "QbloxHardwareCompilationConfig",
        "hardware_description": {
            "cluster_A": {
                "instrument_type": "Cluster",
                "ip": "192.168.1.242",
                "modules": {str(slot): {"instrument_type": kind} for slot, kind in modules.items()},
            }
        },
        "hardware_options": {
            "modulation_frequencies": modulation,
            "output_att": output_att or {},
        },
        "connectivity": {"graph": graph},
    }


CHIP_A_HW = _hw(
    modules={6: "QCM_RF", 8: "QRM_RF"},
    graph=[
        ["cluster_A.module8.complex_output_0", "q1:res"],
        ["q1:res", "cluster_A.module8.complex_input_0"],
        ["cluster_A.module6.complex_output_0", "q1:mw"],
    ],
    modulation={
        "q1:mw-q1.01": {"lo_freq": 3.0e9},
        "q1:res-q1.ro": {"lo_freq": 5.1e9},
    },
    output_att={"q1:mw-q1.01": 8, "q1:res-q1.ro": 42},
)

CHIP_A_DUT = {
    "elements": {
        "q1": {"clock_freqs": {"f01": 2941260109.853855, "f12": 0.0, "readout": 5011860333.485025}}
    }
}


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "hw_config.json").write_text(json.dumps(CHIP_A_HW), encoding="utf-8")
    (tmp_path / "dut_config.json").write_text(json.dumps(CHIP_A_DUT), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- plan
def test_plan_mirrors_the_chipA_config(config_dir):
    """One QCM-RF drive output + one QRM-RF readout output, sequencer 0 on each."""
    plan = cm.build_plan(*cm.load_configs(config_dir))

    assert plan.cluster_name == "cluster_A"
    assert plan.ip == "192.168.1.242"
    assert [(g.slot, g.output, g.module_type) for g in plan.groups] == [
        (6, 0, "QCM_RF"),
        (8, 0, "QRM_RF"),
    ]
    drive, readout = plan.groups
    assert drive.lo_freq == 3.0e9 and drive.output_att == 8
    assert readout.lo_freq == 5.1e9 and readout.output_att == 42
    assert [t.sequencer for t in drive.targets] == [0]
    assert [t.sequencer for t in readout.targets] == [0]


def test_nco_is_clock_minus_lo(config_dir):
    """The scheduler's own convention: NCO = clock_freq - lo_freq."""
    drive, readout = cm.build_plan(*cm.load_configs(config_dir)).groups

    assert drive.targets[0].nco_freq == pytest.approx(2941260109.853855 - 3.0e9)
    assert readout.targets[0].nco_freq == pytest.approx(5011860333.485025 - 5.1e9)
    # and the tone lands back on the qubit / resonator
    assert drive.lo_freq + drive.targets[0].nco_freq == pytest.approx(2941260109.853855)


@pytest.mark.parametrize(
    ("clock", "expected"),
    [("q1.01", 2941260109.853855), ("q1.ro", 5011860333.485025), ("q1.12", 0.0)],
)
def test_clock_name_resolution(clock, expected):
    assert cm.clock_frequency(CHIP_A_DUT, clock) == pytest.approx(expected)


def test_unknown_clock_is_none():
    assert cm.clock_frequency(CHIP_A_DUT, "q9.ro") is None
    assert cm.clock_frequency(CHIP_A_DUT, "q1.wat") is None


def test_explicit_interm_freq_wins_over_the_dut_clock():
    """An ``interm_freq`` in hardware_options is taken as given — no dut lookup."""
    hw = _hw(
        modules={6: "QCM_RF"},
        graph=[["cluster_A.module6.complex_output_0", "q1:mw"]],
        modulation={"q1:mw-q1.01": {"lo_freq": 3.0e9, "interm_freq": 123e6}},
    )
    (group,) = cm.build_plan(hw, CHIP_A_DUT).groups
    assert group.targets[0].nco_freq == 123e6


def test_multiplexed_feedline_is_one_lo_and_two_sequencers():
    """Two readout port-clocks on one output: ONE LO cal, a sideband cal each."""
    hw = _hw(
        modules={8: "QRM_RF"},
        graph=[
            ["cluster_A.module8.complex_output_0", "q1:res"],
            ["cluster_A.module8.complex_output_0", "q2:res"],
        ],
        modulation={
            "q1:res-q1.ro": {"lo_freq": 5.1e9, "interm_freq": -88e6},
            "q2:res-q2.ro": {"lo_freq": 5.1e9, "interm_freq": 51e6},
        },
    )
    (group,) = cm.build_plan(hw, {}).groups

    assert group.lo_freq == 5.1e9
    assert [(t.portclock, t.sequencer, t.nco_freq) for t in group.targets] == [
        ("q1:res-q1.ro", 0, -88e6),
        ("q2:res-q2.ro", 1, 51e6),
    ]


def test_sequencer_override(config_dir):
    hw, dut = cm.load_configs(config_dir)
    plan = cm.build_plan(hw, dut, sequencer_map={"q1:res-q1.ro": 3})
    assert plan.groups[1].targets[0].sequencer == 3


def test_two_los_on_one_output_is_refused():
    hw = _hw(
        modules={8: "QRM_RF"},
        graph=[
            ["cluster_A.module8.complex_output_0", "q1:res"],
            ["cluster_A.module8.complex_output_0", "q2:res"],
        ],
        modulation={
            "q1:res-q1.ro": {"lo_freq": 5.1e9, "interm_freq": 0.0},
            "q2:res-q2.ro": {"lo_freq": 6.0e9, "interm_freq": 0.0},
        },
    )
    with pytest.raises(SystemExit, match="one output has one LO"):
        cm.build_plan(hw, {})


def test_nco_out_of_range_is_refused():
    """A stale LO leaves the IF beyond +/-500 MHz — better a loud refusal than a
    calibration at a frequency the sequencer cannot actually play."""
    hw = _hw(
        modules={6: "QCM_RF"},
        graph=[["cluster_A.module6.complex_output_0", "q1:mw"]],
        modulation={"q1:mw-q1.01": {"lo_freq": 4.0e9}},  # f01 is 2.94 GHz -> -1.06 GHz
    )
    with pytest.raises(SystemExit, match=r"outside the \+/-500 MHz range"):
        cm.build_plan(hw, CHIP_A_DUT)


def test_lo_out_of_range_is_refused():
    hw = _hw(
        modules={6: "QCM_RF"},
        graph=[["cluster_A.module6.complex_output_0", "q1:mw"]],
        modulation={"q1:mw-q1.01": {"lo_freq": 1.0e9, "interm_freq": 0.0}},
    )
    with pytest.raises(SystemExit, match="2-18 GHz"):
        cm.build_plan(hw, {})


def test_baseband_and_input_only_ports_are_skipped(capsys):
    """AMC is RF-modules-only: a baseband QCM has no internal mixer to calibrate,
    and a complex INPUT is not an output."""
    hw = _hw(
        modules={2: "QCM", 8: "QRM_RF"},
        graph=[
            ["cluster_A.module2.complex_output_0", "q1:fl"],
            ["q1:res", "cluster_A.module8.complex_input_0"],
        ],
        modulation={
            "q1:fl-q1.flux": {"lo_freq": 3.0e9, "interm_freq": 0.0},
            "q1:res-q1.ro": {"lo_freq": 5.1e9, "interm_freq": -88e6},
        },
    )
    plan = cm.build_plan(hw, {})
    out = capsys.readouterr().out

    assert plan.groups == []
    assert "not an RF module" in out  # the baseband flux line
    assert "drives no complex output" in out  # readout reached only via the input edge


def test_slot_and_portclock_filters(config_dir):
    hw, dut = cm.load_configs(config_dir)
    assert [g.slot for g in cm.build_plan(hw, dut, slots=[6]).groups] == [6]
    assert [g.slot for g in cm.build_plan(hw, dut, portclocks=["q1:res-q1.ro"]).groups] == [8]


def test_tone_sequence_is_the_tutorial_program():
    seq = cm._tone_sequence(0.3)
    assert seq["waveforms"]["dc"]["data"] == [0.3] * cm.TONE_SAMPLES
    assert "wait_sync" in seq["program"]
    assert f"play    0,0,{cm.TONE_SAMPLES}" in seq["program"]
    assert "jmp     @loop" in seq["program"]


# --------------------------------------------------------------------------- hardware
@pytest.fixture
def dummy_cluster(config_dir):
    """A dummy QCM-RF + QRM-RF cluster matching the chipA config."""
    from qblox_instruments import Cluster, ClusterType
    from qcodes.instrument import Instrument

    try:  # qcodes names are global; free ours without touching anyone else's
        Instrument.find_instrument("amc_test").close()
    except KeyError:
        pass
    cluster = Cluster(
        "amc_test",
        dummy_cfg={6: ClusterType.CLUSTER_QCM_RF, 8: ClusterType.CLUSTER_QRM_RF},
    )
    yield cluster
    cluster.close()


@pytest.fixture
def moved_values(monkeypatch):
    """Make the dummy look like hardware where the calibration actually did something.

    On a dummy the cal calls are no-ops, so every real read comes back on the vendor
    defaults — which is now (correctly) reported as ``no-op`` and never cached. Any test
    about the CALIBRATED path has to fake movement.
    """
    monkeypatch.setattr(
        cm, "_read_lo_state", lambda module, output: {"offset_path0": -8.1, "offset_path1": -9.8}
    )
    monkeypatch.setattr(
        cm,
        "_read_sideband_state",
        lambda sequencer: {"gain_ratio": 0.987, "phase_offset_degree": -4.2},
    )


def test_dummy_run_configures_lo_nco_and_channel_map(dummy_cluster, config_dir):
    plan = cm.build_plan(*cm.load_configs(config_dir))
    cache = {"lo": {}, "sideband": {}, "history": []}

    records = cm.calibrate_cluster(dummy_cluster, plan, cache=cache)

    qcm, qrm = dummy_cluster.modules[5], dummy_cluster.modules[7]
    # QCM-RF: an LO per output; QRM-RF: one LO shared with the input.
    assert qcm.out0_lo_freq() == 3_000_000_000
    assert qrm.out0_in0_lo_freq() == 5_100_000_000
    assert qcm.sequencer0.nco_freq() == pytest.approx(2941260109.853855 - 3.0e9, abs=1)
    assert qrm.sequencer0.nco_freq() == pytest.approx(5011860333.485025 - 5.1e9, abs=1)
    assert qcm.sequencer0.connect_out0() == "IQ"
    assert qrm.sequencer0.connect_out0() == "IQ"
    assert qcm.sequencer0.mod_en_awg() is True
    # attenuation is left exactly as found (0 on a fresh dummy) — this script calibrates,
    # it does not configure; `scqo run` re-pushes output_att from hw_config every run
    assert qcm.out0_att() == 0 and qrm.out0_att() == 0
    # the marker override that gates the RF switch is released again
    assert qcm.sequencer0.marker_ovr_en() is False
    assert qrm.sequencer0.marker_ovr_en() is False

    # nothing moves on a dummy, so every step is a no-op — and none of it is cached
    assert [(r["step"], r["status"]) for r in records] == [
        ("sideband", "no-op"),
        ("lo", "no-op"),
        ("sideband", "no-op"),
        ("lo", "no-op"),
    ]
    assert cache["lo"] == {} and cache["sideband"] == {}


def test_sideband_is_calibrated_before_the_lo(dummy_cluster, config_dir):
    """ORDER IS LOAD-BEARING. Running the LO cal on a sequencer first makes the following
    sideband_cal() return the vendor defaults: chipA 2026-07-27, sideband-only produced
    ratio 1.0331 while the same code with the LO cal ahead of it produced nothing on
    slot 4 and nulled slot 8 to 1.000031."""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])

    records = cm.calibrate_cluster(dummy_cluster, plan)

    assert [r["step"] for r in records] == ["sideband", "lo"]


def test_an_lo_cal_that_resets_the_sideband_is_caught_and_un_cached(
    dummy_cluster, config_dir, monkeypatch, capsys
):
    """The LO cal runs last because it disturbs the sideband state — verify it did not
    also undo the correction we just cached, rather than trusting that it didn't."""
    reads = iter(
        [
            {"gain_ratio": 1.0, "phase_offset_degree": 0.0},  # before the cal
            {"gain_ratio": 0.987, "phase_offset_degree": -4.2},  # after it — good
            {"gain_ratio": 1.0, "phase_offset_degree": 0.0},  # after the LO cal — wiped
        ]
    )
    monkeypatch.setattr(cm, "_read_sideband_state", lambda sequencer: next(reads))
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    cache = {"lo": {}, "sideband": {}, "history": []}

    records = cm.calibrate_cluster(dummy_cluster, plan, cache=cache)

    assert cache["sideband"] == {}  # un-cached, so a re-run retries
    assert [r["status"] for r in records if r["step"] == "sideband"] == ["no-op"]
    assert "the LO calibration reset it back to the defaults" in capsys.readouterr().out


def test_a_calibration_that_changes_nothing_is_reported_and_not_cached(
    dummy_cluster, config_dir, capsys
):
    """The failure that shipped: sideband_cal() returns the vendor defaults, the firmware
    reports success, and the old code cached it — so every later run said 'cached' for a
    calibration that never happened."""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    cache = {"lo": {}, "sideband": {}, "history": []}

    records = cm.calibrate_cluster(dummy_cluster, plan, cache=cache)
    out = capsys.readouterr().out

    assert {r["status"] for r in records} == {"no-op"}
    assert "WARNING: still at the vendor defaults (path0=0, path1=0)" in out
    assert "WARNING: still at the vendor defaults (ratio=1.0, phase=0.0)" in out
    assert cache["lo"] == {} and cache["sideband"] == {}


def test_a_cached_entry_sitting_on_the_defaults_is_a_miss(dummy_cluster, config_dir):
    """Self-heals a cache file poisoned by an earlier silent failure — no need to
    hand-delete mixer_cal.json."""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    group = plan.groups[0]
    cache = {
        "lo": {group.lo_cache_key: {"offset_path0": 0.0, "offset_path1": 0.0}},
        "sideband": {
            group.sideband_cache_key(group.targets[0]): {
                "gain_ratio": 1.0,
                "phase_offset_degree": -0.0,
            }
        },
        "history": [],
    }

    records = cm.calibrate_cluster(dummy_cluster, plan, cache=cache)

    assert {r["status"] for r in records} == {"no-op"}  # re-ran, did not report 'cached'


@pytest.fixture
def tone_spy(monkeypatch):
    """Record the conditions in force each time the tone is actually started."""
    calls = []
    original = cm._play_tone

    def spy(module, sequencer, index, *, marker, **kwargs):
        calls.append({"index": index, "marker": marker, "att": module.out0_att(), **kwargs})
        return original(module, sequencer, index, marker=marker, **kwargs)

    monkeypatch.setattr(cm, "_play_tone", spy)
    return calls


def test_calibration_runs_at_cal_att_and_restores_the_module_value(
    dummy_cluster, config_dir, tone_spy
):
    """The attenuator sits after the mixer, so the operating value can bury the image.
    Calibrate low, put back what the module had."""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])
    qrm = dummy_cluster.modules[7]
    qrm.out0_att(42)

    cm.calibrate_cluster(dummy_cluster, plan, cal_att=0)

    assert tone_spy and {c["att"] for c in tone_spy} == {0}  # 0 dB while the tone plays
    assert qrm.out0_att() == 42  # restored afterwards


def test_cal_att_keep_leaves_attenuation_alone(dummy_cluster, config_dir):
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])
    qrm = dummy_cluster.modules[7]
    qrm.out0_att(30)

    cm.calibrate_cluster(dummy_cluster, plan, cal_att=None)

    assert qrm.out0_att() == 30


def test_attenuation_is_restored_when_a_step_raises(dummy_cluster, config_dir, monkeypatch):
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])
    qrm = dummy_cluster.modules[7]
    qrm.out0_att(42)
    monkeypatch.setattr(
        cm, "_read_lo_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        cm.calibrate_cluster(dummy_cluster, plan, cal_att=0)

    assert qrm.out0_att() == 42


def test_output_switch_stays_open_by_default(dummy_cluster, config_dir, tone_spy):
    """Nothing reaches the fridge while the tone plays: the AMC detector is internal, and
    Qblox opens the switch during the cal anyway. --switch-on is for analyzer work."""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])

    cm.calibrate_cluster(dummy_cluster, plan)
    assert {c["marker"] for c in tone_spy} == {0}

    tone_spy.clear()
    cm.calibrate_cluster(dummy_cluster, plan, switch_on=True)
    assert {c["marker"] for c in tone_spy} == {cm._OUTPUT_MARKER[("QCM_RF", 0)]}


class _FakeStatus:
    def __init__(self, state):
        self.state = state


class _FakeModule:
    def __init__(self, states):
        self.sequencers = [None] * len(states)
        self._states = states

    def get_sequencer_status(self, index):
        state = self._states[index]
        if state is None:
            raise RuntimeError("unreadable")
        return _FakeStatus(f"SequencerStates.{state}")


def test_running_sequencers_snapshot_picks_the_active_states():
    """RUNNING / ARMED / Q1_STOPPED were doing something; IDLE and STOPPED were not.
    A status we cannot read is not restartable, so it is left out rather than guessed."""
    module = _FakeModule(["RUNNING", "IDLE", "ARMED", "STOPPED", "Q1_STOPPED", None])
    assert cm._running_sequencers(module) == [0, 2, 4]


def test_running_sequencers_are_restarted(dummy_cluster, config_dir, monkeypatch):
    """Calibration interrupts every sequencer in the module — the ones that were
    running must come back, or an unrelated experiment silently goes quiet.

    The dummy transport always reports STOPPED, so the snapshot itself is stubbed
    (it is covered above); what this pins down is that whatever it reports gets
    re-armed and restarted after the module is done.
    """
    monkeypatch.setattr(cm, "_running_sequencers", lambda module: [2, 4])
    qcm = dummy_cluster.modules[5]
    calls = []
    for verb in ("arm_sequencer", "start_sequencer"):
        original = getattr(qcm, verb)
        monkeypatch.setattr(
            qcm, verb, lambda index, _v=verb, _o=original: (calls.append((_v, index)), _o(index))[1]
        )

    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    cm.calibrate_cluster(dummy_cluster, plan)

    assert calls[-4:] == [
        ("arm_sequencer", 2),
        ("start_sequencer", 2),
        ("arm_sequencer", 4),
        ("start_sequencer", 4),
    ]


def test_the_sequencer_we_borrowed_is_not_restarted(dummy_cluster, config_dir, monkeypatch, capsys):
    """Its program is now the calibration tone — restarting it would play a steady
    carrier into the fridge forever. Leave it stopped and say so."""
    monkeypatch.setattr(cm, "_running_sequencers", lambda module: [0, 2])
    qcm = dummy_cluster.modules[5]
    started = []
    monkeypatch.setattr(qcm, "start_sequencer", lambda index: started.append(index))

    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    assert plan.groups[0].targets[0].sequencer == 0  # seq0 is the one we borrow
    cm.calibrate_cluster(dummy_cluster, plan)

    # seq0 starts twice during the run itself (the LO tone, then the sideband tone);
    # the restart phase is the tail, and it restarts only the sequencer we never touched.
    assert started[-1] == 2 and started.count(2) == 1
    assert "seq0 was running but now holds the calibration tone" in capsys.readouterr().out


def test_sequencers_are_restarted_even_when_calibration_fails(
    dummy_cluster, config_dir, monkeypatch
):
    """A cal that blows up mid-module must not leave the lab's other sequencers dark."""
    monkeypatch.setattr(cm, "_running_sequencers", lambda module: [3])
    restarted = []
    qcm = dummy_cluster.modules[5]
    monkeypatch.setattr(qcm, "start_sequencer", lambda index: restarted.append(index))
    monkeypatch.setattr(cm, "calibrate_group", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    with pytest.raises(RuntimeError, match="boom"):
        cm.calibrate_cluster(dummy_cluster, plan)

    assert restarted == [3]


def test_cache_serves_the_second_run_and_force_overrides_it(
    dummy_cluster, config_dir, moved_values
):
    plan = cm.build_plan(*cm.load_configs(config_dir))
    cache = {"lo": {}, "sideband": {}, "history": []}

    cm.calibrate_cluster(dummy_cluster, plan, cache=cache)
    assert set(cache["lo"]) == {"slot6/out0/lo3000000000", "slot8/out0/lo5100000000"}
    assert len(cache["sideband"]) == 2

    again = cm.calibrate_cluster(dummy_cluster, plan, cache=cache)
    assert {r["status"] for r in again} == {"cached"}

    forced = cm.calibrate_cluster(dummy_cluster, plan, cache=cache, force=True)
    assert {r["status"] for r in forced} == {"calibrated"}


def test_cache_is_invalidated_when_the_hardware_no_longer_holds_the_values(
    dummy_cluster, config_dir, monkeypatch
):
    """Corrections live in volatile cluster state: a reboot (or anyone else's
    ``.set()``) clears them. The cache is validated against the LIVE values, so
    that shows up as a miss rather than a silently skipped calibration."""
    live = {"offset_path0": -8.1, "offset_path1": -9.8}
    monkeypatch.setattr(cm, "_read_lo_state", lambda module, output: dict(live))
    monkeypatch.setattr(
        cm,
        "_read_sideband_state",
        lambda sequencer: {"gain_ratio": 0.987, "phase_offset_degree": -4.2},
    )
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    cache = {"lo": {}, "sideband": {}, "history": []}
    cm.calibrate_cluster(dummy_cluster, plan, cache=cache)
    assert {r["status"] for r in cm.calibrate_cluster(dummy_cluster, plan, cache=cache)} == {
        "cached"
    }

    live["offset_path0"] = 12.5  # as if the module had been power-cycled and re-set

    records = cm.calibrate_cluster(dummy_cluster, plan, cache=cache)
    assert [(r["step"], r["status"]) for r in records] == [
        ("sideband", "cached"),
        ("lo", "calibrated"),
    ]


def test_lo_only_and_sideband_only(dummy_cluster, config_dir):
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])

    lo = cm.calibrate_cluster(dummy_cluster, plan, do_sideband=False)
    assert [r["step"] for r in lo] == ["lo"]

    sideband = cm.calibrate_cluster(dummy_cluster, plan, do_lo=False)
    assert [r["step"] for r in sideband] == ["sideband"]


# --------------------------------------------------------------- the tone must play
def test_wait_running_is_skipped_on_a_dummy(dummy_cluster):
    """The dummy transport always reports STOPPED — the guard must not turn that into
    a failure, or every CI run breaks."""
    assert cm._wait_running(dummy_cluster.modules[5], 0) is None
    cm._require_running(dummy_cluster.modules[5], 0)  # must not raise


def test_a_sequencer_that_never_runs_aborts_with_the_status(dummy_cluster, config_dir, monkeypatch):
    """Without this the routine fails SILENTLY: LO cal works with no tone at all (DC
    feedthrough), so the sideband cal is the only thing that notices, and it just
    returns the defaults."""
    monkeypatch.setattr(cm, "_wait_running", lambda module, index: "SequencerStates.STOPPED")
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])

    with pytest.raises(SystemExit, match="never reached RUNNING"):
        cm.calibrate_cluster(dummy_cluster, plan)


def test_mixer_corrections_are_reset_to_defaults_before_each_sideband_cal(
    dummy_cluster, config_dir
):
    """Each attempt must start from the vendor defaults, so the AMC measures the raw
    imbalance and "did it move?" is unambiguous — that check is what the retry loop and the
    no-op detection both key off.

    (The write was originally added because it looked like the thing that made slot 4 work
    on 2026-07-27; later runs showed ``sideband_cal()`` is non-deterministic, so treat that
    as unproven. The behaviour is kept because the retry loop genuinely needs it.)"""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    seq = dummy_cluster.modules[5].sequencer0
    seq.mixer_corr_gain_ratio(1.5)
    seq.mixer_corr_phase_offset_degree(20.0)

    cm.calibrate_cluster(dummy_cluster, plan, do_lo=False)

    # a dummy never calibrates, so whatever is left IS what we wrote going in
    assert seq.mixer_corr_gain_ratio() == cm._SIDEBAND_DEFAULTS["gain_ratio"]
    assert seq.mixer_corr_phase_offset_degree() == cm._SIDEBAND_DEFAULTS["phase_offset_degree"]


def test_every_calibration_step_re_arms_its_own_tone(dummy_cluster, config_dir, tone_spy, monkeypatch):
    """Upload once and play twice and the second cal finds nothing: a calibration
    interrupts the module's sequencers, so "stopped then re-armed" is not the same as a
    fresh upload (chipA slot 8, 2026-07-27). Every step gets its own _arm_tone."""
    uploads = []
    original = cm._arm_tone
    monkeypatch.setattr(
        cm,
        "_arm_tone",
        lambda module, group, target, amp: (uploads.append(target.sequencer), original(module, group, target, amp))[1],
    )
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])

    cm.calibrate_cluster(dummy_cluster, plan)

    assert uploads == [0, 0]  # one before the sideband tone, one before the LO tone
    assert len(tone_spy) == 2


def test_sequencer_flags_are_not_cleared(dummy_cluster, config_dir, monkeypatch):
    """clear_sequencer_flags() BREAKS the calibration on chipA — --diagnose ran identical
    conditions with and without it (trials D vs E, 2026-07-27) and only the run without it
    produced a correction. The production path must never call it."""
    cleared = []
    qcm = dummy_cluster.modules[5]
    monkeypatch.setattr(qcm, "clear_sequencer_flags", lambda index: cleared.append(index))

    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[6])
    cm.calibrate_cluster(dummy_cluster, plan)

    assert cleared == []


def test_diagnose_still_probes_flag_clearing(dummy_cluster, config_dir, monkeypatch):
    """It stays in the matrix as trial E so the finding stays falsifiable on other chips."""
    cleared = []
    qrm = dummy_cluster.modules[7]
    monkeypatch.setattr(qrm, "clear_sequencer_flags", lambda index: cleared.append(index))

    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])
    cm.diagnose(dummy_cluster, plan)

    assert cleared == [0]  # exactly one trial (E) clears flags
    assert [t[4] for t in cm.DIAGNOSE_TRIALS] == [False, False, False, False, True]


# --------------------------------------------------------------------------- diagnose
def test_diagnose_runs_every_trial_and_restores_the_module(dummy_cluster, config_dir):
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])
    qrm = dummy_cluster.modules[7]
    qrm.out0_att(42)

    results = cm.diagnose(dummy_cluster, plan)

    assert [r["trial"] for r in results] == [t[0] for t in cm.DIAGNOSE_TRIALS]
    assert [r["att"] for r in results] == [42, 0, 0, 0, 0]  # trial A holds the config value
    assert not any(r["moved"] for r in results)  # a dummy can never move
    assert qrm.out0_att() == 42  # restored


def test_diagnose_reports_when_the_tone_never_played(dummy_cluster, config_dir, monkeypatch, capsys):
    """The verdict must point at the sequencer, not at the attenuation, when the tone
    never played — that is the difference between a fixable knob and a dead sequencer."""
    monkeypatch.setattr(cm, "_wait_running", lambda module, index: "SequencerStates.ARMED")
    monkeypatch.setattr(type(dummy_cluster.modules[7]), "is_dummy", property(lambda self: False))
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])

    cm.diagnose(dummy_cluster, plan)

    out = capsys.readouterr().out
    assert "no trial produced a correction" in out
    assert "the tone is not playing at all" in out


def test_diagnose_says_so_on_a_dummy(dummy_cluster, config_dir, capsys):
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])

    cm.diagnose(dummy_cluster, plan)

    assert "dummy cluster -- the cal calls are no-ops" in capsys.readouterr().out


def test_diagnose_names_the_first_working_trial(dummy_cluster, config_dir, monkeypatch, capsys):
    """Simulate 'it only works once the attenuation comes down' — trial B."""
    seen = {"n": 0}

    def fake_read(sequencer):
        seen["n"] += 1
        if seen["n"] <= 1:  # trial A reads defaults, everything after is corrected
            return {"gain_ratio": 1.0, "phase_offset_degree": 0.0}
        return {"gain_ratio": 0.987, "phase_offset_degree": -4.2}

    monkeypatch.setattr(cm, "_read_sideband_state", fake_read)
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])

    results = cm.diagnose(dummy_cluster, plan)

    assert [r["moved"] for r in results] == [False, True, True, True, True]
    assert "VERDICT: B is the first trial that works" in capsys.readouterr().out


def test_cli_diagnose_writes_no_cache(config_dir, capsys):
    assert cm.main([str(config_dir), "--diagnose", "--dummy"]) == 0

    assert "diagnosing slot" in capsys.readouterr().out
    assert not (config_dir / "mixer_cal.json").exists()


def test_module_type_mismatch_is_refused(dummy_cluster, config_dir):
    """The dummy has QRM-RF in slot 8; claim a QCM-RF and the run must stop rather
    than calibrate the wrong output."""
    plan = cm.build_plan(*cm.load_configs(config_dir), slots=[8])
    plan.groups[0].module_type = "QCM_RF"

    with pytest.raises(SystemExit, match="the cluster reports QRM_RF"):
        cm.calibrate_cluster(dummy_cluster, plan)


def test_cli_dry_run_touches_no_hardware(config_dir, capsys):
    assert cm.main([str(config_dir), "--dry-run"]) == 0
    out = capsys.readouterr().out

    assert "slot 6 QCM_RF out0: LO 3.000000 GHz" in out
    assert "slot 8 QRM_RF out0: LO 5.100000 GHz" in out
    assert not (config_dir / "mixer_cal.json").exists()


def test_cli_dummy_writes_history_but_caches_nothing(config_dir, capsys):
    """A dummy can only ever produce no-ops. History still records them (that IS the
    drift log), but nothing is cached and the run is reported honestly."""
    assert cm.main([str(config_dir), "--dummy"]) == 0

    cache = json.loads((config_dir / "mixer_cal.json").read_text(encoding="utf-8"))
    assert cache["lo"] == {} and cache["sideband"] == {}
    assert len(cache["history"]) == 1
    entry = cache["history"][0]
    assert entry["dummy"] is True and entry["amp"] == cm.DEFAULT_IF_AMP
    assert entry["cal_att"] == 0 and entry["switch_on"] is False
    assert len(entry["records"]) == 4
    assert "0 calibrated, 0 cached, 4 no-op" in capsys.readouterr().out


def test_cli_exits_non_zero_on_a_real_no_op(config_dir, monkeypatch, capsys):
    """On hardware a no-op is a failure, not a shrug — it must be visible to a shell."""
    monkeypatch.setattr(cm, "open_cluster", _fake_open_cluster(config_dir))

    assert cm.main([str(config_dir), "--no-cache-write"]) == 1
    assert "FAILED: the calibration ran but changed nothing" in capsys.readouterr().out


def _fake_open_cluster(config_dir):
    """A dummy cluster that does NOT announce itself as one, so main() takes the
    hardware branch (exit codes, the FAILED banner)."""

    def opener(plan, *, ip, dummy, name="mixercal"):
        from qblox_instruments import Cluster, ClusterType
        from qcodes.instrument import Instrument

        try:
            Instrument.find_instrument("amc_fake").close()
        except KeyError:
            pass
        return Cluster(
            "amc_fake",
            dummy_cfg={6: ClusterType.CLUSTER_QCM_RF, 8: ClusterType.CLUSTER_QRM_RF},
        )

    return opener


def test_cli_rejects_contradictory_and_out_of_range_flags(config_dir):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        cm.main([str(config_dir), "--lo-only", "--sideband-only"])
    with pytest.raises(SystemExit, match="--amp must be in"):
        cm.main([str(config_dir), "--amp", "1.5"])
    with pytest.raises(SystemExit, match="--cal-att expects"):
        cm.main([str(config_dir), "--cal-att", "7"])  # attenuator steps in 2 dB
    with pytest.raises(SystemExit, match="--cal-att expects"):
        cm.main([str(config_dir), "--cal-att", "low"])


def test_cli_cal_att_keep(config_dir):
    assert cm.main([str(config_dir), "--dummy", "--cal-att", "keep", "--no-cache-write"]) == 0


def test_the_precondition_reaches_the_operator_before_they_start():
    """`calibrate_mixers.py` prints its PRECONDITIONS block at RUNTIME - i.e.
    after the operator has already started it. `scqo state --fields` carries the
    same sentence in the command inventory, which is where they can read it
    BEFORE. Pinned literally so the two cannot drift apart."""
    from pathlib import Path

    from scqo_qblox.backend.fieldmap import OPERATOR_COMMANDS

    phrase = "no scqo session / HardwareAgent holding"
    entry = next(c for c in OPERATOR_COMMANDS if c.name == "calibrate_mixers")
    assert phrase in entry.caution
    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_mixers.py"
    assert phrase in script.read_text(encoding="utf-8")
