"""`use_state_discrimination` end to end on Qblox: knobs, probes, decode, proposal.

The feature spans four places that must agree, and three of the four fail SILENTLY
if they disagree:

* the two knobs (`readout_rotation_rad` / `readout_threshold`) must land on
  `element.measure.acq_rotation` / `acq_threshold`, with the rad->deg conversion
  AND the sign flip (the vendors are opposite-handed) — a missed conversion is a
  factor of 57 in the rotation, a missed flip mirrors the frame (chipA 2026-08-08:
  every shot classified |g>), and either just gives a bad fit;
* the probes must ask for `ThresholdedAcquisition`;
* `_to_canonical` must emit `state`, not `I`/`Q` — a thresholded result is REAL, so
  the old unpack would put the population in `I`, leave `Q` at zero, and still pass
  the `("I","Q")` contract;
* `single_shot_readout.update()` must propose the two knobs, or nothing ever
  calibrates them and the guard refuses every discriminated run.

The one loud failure is the guard, which is deliberate: refusing an uncalibrated
run by name beats discriminating every shot against zero.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

#: the four experiments carrying StateReadoutParameters that have a Qblox probe
#: (qubit_sqrb / the flux coherence pair carry it too, but are QM-only)
CARRIERS = ["qubit_relaxation", "qubit_ramsey", "qubit_echo", "qubit_power_rabi"]

#: small but legal for every carrier's own Parameters minimums. Two point-count
#: spellings because the dict is filtered per experiment: `num_amp_points` is the
#: amplitude capability's, `num_points` the time-sweep carriers'.
SMALL = {"num_points": 5, "num_amp_points": 5, "num_averages": 2,
         "end_amp_factor": 0.5}

#: a plausible calibrated discriminator (rotation in RADIANS, the neutral unit).
#: NEGATIVE on purpose — that is what the solve returns in practice, and the
#: sequencer refuses negative degrees, so every compile test below is also a
#: regression for the folding (chipA 2026-07-26). A positive fixture here is what
#: let that bug reach hardware.
ROTATION_RAD = -0.38962897776554817
THRESHOLD = -3.25e-4


def _calibrate(exp, target="q1"):
    """Arm the discriminator the way `scqo accept` would — through the RECORDING
    device, which is also what the probes read. Writing the element directly would
    not do: the recording surface seeds its cache at construction, so a late vendor
    write is invisible to `measure_kwargs` (and to the guard)."""
    view = exp.device.channel(target, "readout")
    view.readout_rotation_rad = ROTATION_RAD
    view.readout_threshold = THRESHOLD
    return view


def _acq_protocols(schedule) -> set:
    """Every Measure's requested acq_protocol, from anywhere in the schedule tree.

    The probes nest their Measures two loops deep, so this has to descend both
    subschedules (``.operations``) and loop bodies (``LoopOperation.body``)."""
    found: set = set()
    ops = getattr(schedule, "operations", None)
    for op in (ops.values() if ops else ()):
        if hasattr(op, "body"):
            found |= _acq_protocols(op.body)
        elif hasattr(op, "operations"):
            found |= _acq_protocols(op)
        else:
            gate = getattr(op, "data", {}).get("gate_info") or {}
            if gate.get("operation_type") == "measure":
                found.add(gate.get("acq_protocol"))
    return found


def _experiment(tmp_path, roster, name, *, use_state):
    from scqo.experiments import get

    import scqo_qblox.experiments  # noqa: F401  (registers the qblox probes)

    backend = make_backend(tmp_path, roster)
    cls = get(name)
    params = {k: v for k, v in SMALL.items() if k in cls.Parameters.model_fields}
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], use_state_discrimination=use_state,
                                         **params))
    return backend, exp


# --------------------------------------------------------------- the two knobs

def test_rotation_is_stored_in_degrees_on_the_element(tmp_path, roster):
    """The neutral field is radians in QM's integration_weights_angle convention;
    the vendor knob is degrees the OTHER way round (acq_rotation turns the data
    counterclockwise, a QM weights angle turns it clockwise). The conversion AND
    the negation live at this one boundary — without both, the same `scqo set`
    means two different rotations per backend."""
    backend = make_backend(tmp_path, roster)
    view = backend.device.component("fl.q1")

    view.readout_rotation_rad = math.pi / 4
    element = backend.device.component("fl.q1")._element
    assert float(element.measure.acq_rotation) == pytest.approx(315.0)
    assert view.readout_rotation_rad == pytest.approx(math.pi / 4)


def test_a_negative_rotation_is_folded_into_the_vendor_range(tmp_path, roster):
    """The sequencer takes degrees in [0, 360] and REFUSES anything outside
    (constants.MIN/MAX_PHASE_ROTATION_ACQ). The solve routinely returns a negative
    angle, so the view folds — chipA 2026-07-26 otherwise died at compile time with
    "Attempting to configure acq_rotation to -21.712643880286674 ... while the
    hardware requires it to be between 0 and 360"."""
    from qblox_scheduler.backends.qblox import constants

    backend = make_backend(tmp_path, roster)
    view = backend.device.component("fl.q1")
    element = backend.device.component("fl.q1")._element

    for value in (-0.38962897776554817, -3.0, 0.0, math.pi, 7.5, -12.0):
        view.readout_rotation_rad = value
        deg = float(element.measure.acq_rotation)
        assert constants.MIN_PHASE_ROTATION_ACQ <= deg <= constants.MAX_PHASE_ROTATION_ACQ, (
            f"{value} rad -> {deg} deg is outside the sequencer's range")
        # the same physical angle comes back, in the neutral (-pi, pi] convention.
        # The bound is |x| <= pi rather than a half-open interval: pi and -pi ARE
        # the same angle, and the 1e-12 quantization can push the boundary case to
        # either side of it.
        assert abs(view.readout_rotation_rad) <= math.pi + 1e-9
        assert math.isclose(math.cos(view.readout_rotation_rad), math.cos(value), abs_tol=1e-9)
        assert math.isclose(math.sin(view.readout_rotation_rad), math.sin(value), abs_tol=1e-9)


def test_threshold_needs_no_folding(tmp_path, roster):
    """The other half of the range story: acq_threshold's limits are +-1.7e7, so a
    real solved threshold (~1e-4) never approaches them. Only the rotation folds."""
    from qblox_scheduler.backends.qblox import constants

    backend = make_backend(tmp_path, roster)
    view = backend.device.component("fl.q1")

    view.readout_threshold = THRESHOLD
    element = backend.device.component("fl.q1")._element
    assert (constants.MIN_DISCRETIZATION_THRESHOLD_ACQ
            <= float(element.measure.acq_threshold)
            <= constants.MAX_DISCRETIZATION_THRESHOLD_ACQ)


def test_threshold_round_trips_unconverted(tmp_path, roster):
    """No conversion: the threshold is compared in the same normalized frame the
    probes acquire in (the compiler multiplies by integration_length; an
    SSBIntegrationComplex result is divided by acq_duration)."""
    backend = make_backend(tmp_path, roster)
    view = backend.device.component("fl.q1")

    view.readout_threshold = THRESHOLD
    element = backend.device.component("fl.q1")._element
    assert float(element.measure.acq_threshold) == pytest.approx(THRESHOLD)
    assert view.readout_threshold == pytest.approx(THRESHOLD)


def test_rus_threshold_is_still_unrealized(tmp_path, roster):
    """Repeat-until-success is QM-only — there is no acq_rus knob to write."""
    backend = make_backend(tmp_path, roster)
    view = backend.device.component("fl.q1")

    with pytest.raises(NotImplementedError, match="rus"):
        view.readout_rus_threshold = -1e-4
    with pytest.raises(NotImplementedError, match="rus"):
        _ = view.readout_rus_threshold


def test_the_fieldmap_agrees_with_the_view():
    """The drift alarm in miniature: the two promoted fields must be bindings and
    NOT unrealized, and the rus field the other way round."""
    from scqo_qblox.backend.fieldmap import FIELD_BINDINGS, UNREALIZED, VENDOR_ONLY

    readout = FIELD_BINDINGS["readout"]
    assert "readout_rotation_rad" in readout and "readout_threshold" in readout
    assert set(UNREALIZED["readout"]) == {"readout_rus_threshold"}
    # the VENDOR_ONLY "would realize" placeholders are gone — they described the
    # unrealized state and would now contradict the bindings
    assert "acq_rotation" not in VENDOR_ONLY and "acq_threshold" not in VENDOR_ONLY


# ------------------------------------------------------------------ the probes

@pytest.mark.parametrize("name", CARRIERS)
def test_carrier_compiles_discriminated(tmp_path, roster, name):
    """The whole point: with the discriminator armed, each carrier COMPILES in
    state mode — the schedule the cluster would actually run."""
    backend, exp = _experiment(tmp_path, roster, name, use_state=True)
    _calibrate(exp)
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0

    compile_probe(backend, exp)


@pytest.mark.parametrize("name", CARRIERS)
def test_carrier_asks_for_the_thresholded_protocol(tmp_path, roster, name):
    """Compiling proves it is legal; this proves it is DISCRIMINATED. Without the
    protocol the run silently returns averaged I/Q and the flag means nothing."""
    _backend, exp = _experiment(tmp_path, roster, name, use_state=True)
    _calibrate(exp)
    exp.sweep_axes = exp.define_sweep()

    assert _acq_protocols(exp.probe()) == {"ThresholdedAcquisition"}


@pytest.mark.parametrize("name", CARRIERS)
def test_carrier_stays_averaged_by_default(tmp_path, roster, name):
    """The default is unchanged — no probe silently acquires a protocol it did not
    before, and an uncalibrated element is fine as long as the flag is off.
    ``None`` means "let the device config choose", i.e. SSBIntegrationComplex."""
    _backend, exp = _experiment(tmp_path, roster, name, use_state=False)
    exp.sweep_axes = exp.define_sweep()

    assert _acq_protocols(exp.probe()) == {None}


def test_uncalibrated_discrimination_is_refused_by_name(tmp_path, roster):
    """The guard QM does not have. On QM an uncalibrated threshold is None and dies
    as a TypeError deep inside the QUA DSL, naming nothing."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation", use_state=True)
    exp.sweep_axes = exp.define_sweep()

    with pytest.raises(ValueError, match="single_shot_readout"):
        exp.probe()


# ------------------------------------------------------------------ the decode

def _raw(values, *, protocol, key="S_21_q1"):
    """A raw hardware dataset shaped the way qblox_scheduler returns one."""
    arr = xr.DataArray(np.asarray(values), dims=("acq_index",))
    arr.attrs["acq_protocol"] = protocol
    return xr.Dataset({key: arr})


def test_thresholded_average_becomes_a_population_variable(tmp_path, roster):
    """The silent-failure guard: the vendor's own acq_protocol attr decides, not the
    parameter, because the attr describes what the hardware actually did. With no
    shot_idx sweep the cluster AVERAGED the thresholded shots, so under the readout
    schema the values are probabilities and land on `population` — `state` is
    reserved for per-shot outcomes."""
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    _backend, exp = _experiment(tmp_path, roster, "qubit_relaxation", use_state=True)
    exp.sweep_axes = exp.define_sweep()
    n = exp.sweep_axes["wait_time_ns"].size
    populations = np.linspace(0.9, 0.1, n)

    ds = QbloxBackend._to_canonical(_raw(populations, protocol="ThresholdedAcquisition"), exp)

    assert set(ds.data_vars) == {"population"}
    assert ds["population"].dims == ("target", "wait_time_ns")
    assert list(ds.coords["target"].values) == ["q1"]
    np.testing.assert_allclose(ds["population"].values[0], populations)
    exp.Contract.validate(ds)  # the POPULATION_ALT branch of the neutral contract


def test_thresholded_shot_sweep_keeps_the_state_variable(tmp_path, roster):
    """With a shot_idx sweep every thresholded value is one shot's OUTCOME, so the
    variable stays `state` (the per-shot form) — the parity monitors' telegraph."""
    from conftest import make_backend, make_experiment

    from scqo_qblox.backend.qblox_backend import QbloxBackend
    from scqo_qblox.experiments.qubit_parity_switch_continuous import (
        QbloxQubitParitySwitchContinuous,
    )

    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        QbloxQubitParitySwitchContinuous, backend, roster,
        QbloxQubitParitySwitchContinuous.Parameters(
            targets=["q1"], num_shots=100, use_state_discrimination=True))
    # define_sweep derives the fixed idle from the stored splitting and the shot
    # period from the depletion knob (the test_parity_switch fixture values).
    exp.device.channel("q1", "drive").parity_delta_f_hz = 250e3
    exp.device.channel("q1", "readout").readout_depletion_s = 795.77e-9
    exp.sweep_axes = exp.define_sweep()
    n = exp.sweep_axes["shot_idx"].size
    shots = (np.arange(n) % 2).astype(float)

    ds = QbloxBackend._to_canonical(_raw(shots, protocol="ThresholdedAcquisition"), exp)

    assert set(ds.data_vars) == {"state"}
    assert ds["state"].dims == ("target", "shot_idx")
    exp.Contract.validate(ds)


def test_thresholded_discrete_decode_keeps_state_with_meas_axis(tmp_path, roster):
    """The discrete variant's two bins per cycle come back as flat labeled bins
    (cycle-major, measurement-minor — the loop order); the decode must fold
    them onto (shot_idx, meas_idx) and still land on per-shot `state`."""
    from conftest import make_backend, make_experiment

    from scqo_qblox.backend.qblox_backend import QbloxBackend
    from scqo_qblox.experiments.qubit_parity_switch_discrete import (
        QbloxQubitParitySwitchDiscrete,
    )

    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        QbloxQubitParitySwitchDiscrete, backend, roster,
        QbloxQubitParitySwitchDiscrete.Parameters(
            targets=["q1"], num_shots=100, use_state_discrimination=True))
    exp.device.channel("q1", "drive").parity_delta_f_hz = 250e3
    exp.device.channel("q1", "readout").readout_depletion_s = 795.77e-9
    exp.sweep_axes = exp.define_sweep()
    n = exp.sweep_axes["shot_idx"].size
    assert exp.sweep_axes["meas_idx"].size == 2
    # m1 alternates per cycle, m2 = NOT m1 -> flat (m1[0], m2[0], m1[1], ...)
    m1 = (np.arange(n) % 2).astype(float)
    flat = np.stack([m1, 1.0 - m1], axis=-1).ravel()

    ds = QbloxBackend._to_canonical(_raw(flat, protocol="ThresholdedAcquisition"), exp)

    assert set(ds.data_vars) == {"state"}
    assert ds["state"].dims == ("target", "shot_idx", "meas_idx")
    np.testing.assert_allclose(ds["state"].values[0, :, 0], m1)
    np.testing.assert_allclose(ds["state"].values[0, :, 1], 1.0 - m1)
    exp.Contract.validate(ds)


def test_the_nan_sentinel_becomes_nan(tmp_path, roster):
    """The compiler substitutes -1 for a NaN threshold result. Left alone that reads
    as a valid population far below the floor and quietly drags the fit."""
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    _backend, exp = _experiment(tmp_path, roster, "qubit_relaxation", use_state=True)
    exp.sweep_axes = exp.define_sweep()
    n = exp.sweep_axes["wait_time_ns"].size
    populations = np.full(n, 0.5)
    populations[2] = -1.0

    ds = QbloxBackend._to_canonical(_raw(populations, protocol="ThresholdedAcquisition"), exp)

    assert np.isnan(ds["population"].values[0, 2])
    assert np.isfinite(ds["population"].values[0, [0, 1, 3]]).all()


def test_untresholded_raw_still_becomes_iq(tmp_path, roster):
    """The default path is untouched."""
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    _backend, exp = _experiment(tmp_path, roster, "qubit_relaxation", use_state=False)
    exp.sweep_axes = exp.define_sweep()
    n = exp.sweep_axes["wait_time_ns"].size
    s21 = np.linspace(1, 2, n) + 1j * np.linspace(-1, 1, n)

    ds = QbloxBackend._to_canonical(_raw(s21, protocol="SSBIntegrationComplex"), exp)

    assert set(ds.data_vars) == {"I", "Q"}
    np.testing.assert_allclose(ds["I"].values[0], s21.real)
    np.testing.assert_allclose(ds["Q"].values[0], s21.imag)


# --------------------------------------------------------------- the proposal

def test_single_shot_readout_proposes_the_discriminator(tmp_path, roster):
    """Closing the loop: the run that measures the blobs is the run that arms the
    two knobs. Without this override nothing on Qblox ever calibrates them."""
    from scqo.experiments import get

    import scqo_qblox.experiments  # noqa: F401

    backend = make_backend(tmp_path, roster)
    cls = get("single_shot_readout")
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_shots=400))
    exp.sweep_axes = exp.define_sweep()

    # two well-separated blobs, |e> offset along +Q so the solved rotation is real
    rng = np.random.default_rng(7)
    n = exp.sweep_axes["shot_idx"].size
    i = np.stack([rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)])
    q = np.stack([rng.normal(0.0, 1.0, n), rng.normal(6.0, 1.0, n)])
    exp.dataset = xr.Dataset(
        {"I": (("target", "prepared_state", "shot_idx"), i[None]),
         "Q": (("target", "prepared_state", "shot_idx"), q[None])},
        coords={"target": ["q1"], **exp.sweep_axes},
    )
    exp.result = exp.estimate()
    assert exp.result.outcomes["q1"].value == "successful"

    before = exp.device.channel("q1", "readout").readout_rotation_rad
    exp.update()

    view = exp.device.channel("q1", "readout")
    assert view.readout_threshold != 0.0, "threshold never proposed"
    assert view.readout_rotation_rad != before, "rotation never proposed"

    # What the rotation must DO, rather than what number it is: the solve returns
    # an angle modulo 2*pi (here ~3*pi/2, i.e. -pi/2), so comparing the value to a
    # quarter turn proves nothing. Rotate the raw blobs by the proposed angle and
    # check the physics — |e> lands ABOVE |g> on I, and the threshold separates them.
    angle = -view.readout_rotation_rad  # the stored field is the absolute -delta
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    g_rot = float(np.mean(i[0] * cos_a - q[0] * sin_a))
    e_rot = float(np.mean(i[1] * cos_a - q[1] * sin_a))
    assert e_rot > g_rot, "the rotation must put |e> above |g> on the I axis"
    assert g_rot < view.readout_threshold < e_rot


def test_single_shot_readout_leaves_rus_alone(tmp_path, roster):
    """readout_rus_threshold is Unrealized here — proposing it would raise, which is
    why the Qblox override writes only two of QM's three."""
    from scqo.experiments import get

    import scqo_qblox.experiments  # noqa: F401

    backend = make_backend(tmp_path, roster)
    cls = get("single_shot_readout")
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_shots=200))
    exp.sweep_axes = exp.define_sweep()
    rng = np.random.default_rng(11)
    n = exp.sweep_axes["shot_idx"].size
    i = np.stack([rng.normal(0.0, 1.0, n), rng.normal(5.0, 1.0, n)])
    q = np.stack([rng.normal(0.0, 1.0, n), rng.normal(0.0, 1.0, n)])
    exp.dataset = xr.Dataset(
        {"I": (("target", "prepared_state", "shot_idx"), i[None]),
         "Q": (("target", "prepared_state", "shot_idx"), q[None])},
        coords={"target": ["q1"], **exp.sweep_axes},
    )
    exp.result = exp.estimate()

    exp.update()  # must not raise NotImplementedError on the rus knob


# ------------------------------------------------- the proposal must CONVERGE

#: the |g>/|e> centres actually measured on chipA by run
#: 20260726-192206-chipA-single_shot_readout-01 (result.json), and the rotation
#: the store held going into it.
CHIPA_G = (-0.00026595257394077356, -6.132655715110543e-05)
CHIPA_E = (0.00016594330254432433, -0.00023867218995908445)
CHIPA_STANDING_ROTATION = -1.177144608928629
#: what the raw blobs ask for — atan2(dQ, dI) of the two centres above
CHIPA_TRUE_ROTATION = -0.38962897776554817


def _ssro_on_blobs(tmp_path, roster, g, e, *, seed=5, sigma=7.36e-05, shots=2000,
                   backend=None):
    """A single_shot_readout sitting on two Gaussian blobs at the given centres,
    already estimated — ready for update()."""
    from scqo.experiments import get

    import scqo_qblox.experiments  # noqa: F401

    backend = backend if backend is not None else make_backend(tmp_path, roster)
    cls = get("single_shot_readout")
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_shots=shots))
    exp.sweep_axes = exp.define_sweep()
    rng = np.random.default_rng(seed)
    n = exp.sweep_axes["shot_idx"].size
    i = np.stack([g[0] + rng.normal(0, sigma, n), e[0] + rng.normal(0, sigma, n)])
    q = np.stack([g[1] + rng.normal(0, sigma, n), e[1] + rng.normal(0, sigma, n)])
    exp.dataset = xr.Dataset(
        {"I": (("target", "prepared_state", "shot_idx"), i[None]),
         "Q": (("target", "prepared_state", "shot_idx"), q[None])},
        coords={"target": ["q1"], **exp.sweep_axes},
    )
    exp.result = exp.estimate()
    return backend, exp


def test_the_proposal_is_idempotent(tmp_path, roster):
    """THE regression. Accepting a proposal and re-measuring the SAME blobs must
    propose the SAME rotation.

    chipA 2026-07-26 violated this three runs running: the Qblox override had
    inherited QM's `current - delta` accumulate rule, which only converges when the
    vendor's rotation feeds back into the acquired data. Qblox's `acq_rotation` is
    read on the thresholded path ONLY, so the blobs come back in the raw frame
    forever, the same delta is measured every run, and the knob walked
    0 -> -0.388 -> -0.788 -> -1.177 rad with no fixed point.
    """
    backend, exp = _ssro_on_blobs(tmp_path, roster, CHIPA_G, CHIPA_E)
    exp.update()
    first = exp.device.channel("q1", "readout").readout_rotation_rad

    # `scqo accept` — the write above already pushed to the element; a fresh
    # experiment re-seeds its recording device from the vendor, exactly as the
    # next run's Session would.
    _backend, exp2 = _ssro_on_blobs(tmp_path, roster, CHIPA_G, CHIPA_E, backend=backend)
    assert exp2.device.channel("q1", "readout").readout_rotation_rad == pytest.approx(first)
    exp2.update()
    second = exp2.device.channel("q1", "readout").readout_rotation_rad

    assert second == pytest.approx(first, abs=1e-12), (
        f"the proposal moved on a re-measurement of identical blobs "
        f"({first} -> {second}): the rotation is accumulating again")


def test_the_chipA_runaway_cannot_return(tmp_path, roster):
    """The same check against the real numbers, with the standing (over-rotated)
    value in place: the proposal must be what the blobs ask for, independent of
    what the knob already holds."""
    backend, exp = _ssro_on_blobs(tmp_path, roster, CHIPA_G, CHIPA_E)
    exp.device.channel("q1", "readout").readout_rotation_rad = CHIPA_STANDING_ROTATION

    exp.update()
    proposed = exp.device.channel("q1", "readout").readout_rotation_rad

    # exactly the angle of the blobs the estimator FITTED (the GMM re-fits the
    # centres from the shots, so pinning the specified centres would be pinning
    # this seed's shot noise)
    fit = exp.result.fit["q1"]
    fitted_angle = math.atan2(fit["mean_e_q"] - fit["mean_g_q"],
                              fit["mean_e_i"] - fit["mean_g_i"])
    assert proposed == pytest.approx(fitted_angle, abs=1e-9)
    # ...and that angle is the one chipA really measured, to within shot noise
    assert proposed == pytest.approx(CHIPA_TRUE_ROTATION, abs=0.02)

    # the accumulate rule would have produced standing - delta = -1.567: a full
    # standing rotation away from the truth, which is the whole bug
    accumulated = CHIPA_STANDING_ROTATION + proposed
    assert abs(proposed - accumulated) > 1.0


# ------------------------------------------- the vendor's rotation handedness

#: the |g>/|e> centres run 20260808-123042-526-chipA-single_shot_readout-01
#: actually measured (its accepted pos_* suggestions). The solved angle here is
#: ~2.15 rad — large enough that a mirrored rotation moves BOTH centres to the
#: same side of the threshold, which is how the sign bug surfaced: the
#: discriminated qubit_relaxation 20260808-123326-971 read ~0 population at
#: every wait while its averaged twin 20260808-123214-853 fitted T1 = 273 us.
#: (The 2026-07 validation angle was only -0.39 rad, where the mirror costs
#: fidelity but keeps the centres on opposite sides — a silent pass.)
CHIPA_0808_G = (1.4181183459582142e-04, -3.570687705820789e-06)
CHIPA_0808_E = (-1.0898111433014777e-04, 3.787100019975846e-04)


def test_the_element_rotation_is_the_vendors_handedness(tmp_path, roster):
    """The number pushed to acq_rotation must work under the VENDOR's convention,
    not the view's inverse of it. Per the qblox_scheduler acquisitions tutorial,
    acq_rotation turns the threshold line CLOCKWISE, so the sequencer's decision
    value for a shot z = I + iQ is Re(z * exp(+i*radians(acq_rotation))) — the
    data turned counterclockwise (cal16 writes degrees(mod(-angle(e-g), 2pi))
    straight to the knob). Model the hardware exactly that way, straight off the
    ELEMENT, and |e> must land above acq_threshold with |g> below it. The view's
    own read-back cannot stand in for this check: it inverts whatever the setter
    did, so it round-trips equally well with BOTH signs."""
    backend, exp = _ssro_on_blobs(tmp_path, roster, CHIPA_0808_G, CHIPA_0808_E)
    exp.update()

    element = backend.device.component("fl.q1")._element
    theta = math.radians(float(element.measure.acq_rotation))
    threshold = float(element.measure.acq_threshold)

    def decision(centre):
        return centre[0] * math.cos(theta) - centre[1] * math.sin(theta)

    assert decision(CHIPA_0808_E) > threshold > decision(CHIPA_0808_G), (
        f"acq_rotation={math.degrees(theta):.2f} deg puts the centres at "
        f"g={decision(CHIPA_0808_G):.3g}, e={decision(CHIPA_0808_E):.3g} against "
        f"acq_threshold={threshold:.3g} under the vendor's clockwise-threshold-"
        f"line convention — the 2026-08-08 sign bug (every shot classified |g>)")


def test_rotation_round_trip_is_a_fixed_point(tmp_path, roster):
    """`radians(degrees(x)) != x` in general, and RecordingDevice._sync_coupled
    compares vendor read-back to its cache with EXACT equality — so a lossy
    round trip emitted a junk `coupled_to` history row on every write to any other
    readout knob (chipA 19:22:43: -1.177144608928629 -> -1.1771446089286293)."""
    backend = make_backend(tmp_path, roster)
    view = backend.device.component("fl.q1")

    for value in (CHIPA_STANDING_ROTATION, CHIPA_TRUE_ROTATION, 0.7853981633974483, -3.0):
        view.readout_rotation_rad = value
        once = view.readout_rotation_rad
        view.readout_rotation_rad = once
        assert view.readout_rotation_rad == once, f"not a fixed point at {value}"
