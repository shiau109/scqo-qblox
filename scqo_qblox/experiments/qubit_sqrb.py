"""Qblox Single Qubit Randomized Benchmarking (SQRB) acquisition probe.

Generates random Clifford sequences of swept depths and computes in-place
recovery gates using the single-qubit Clifford group Cayley closure.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitSQRB``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitSQRB

from ._reset import add_reset
from ._state import measure_kwargs

# Decompositions of the 24 single-qubit Clifford group elements into (theta_deg, phi_deg) rotations
CLIFFORD_PULSES: list[list[tuple[float, float]]] = [
    [],                                                          # 0: I
    [(180.0, 0.0)],                                             # 1: X180
    [(180.0, 90.0)],                                            # 2: Y180
    [(180.0, 90.0), (180.0, 0.0)],                              # 3: Y180, X180
    [(90.0, 0.0), (90.0, 90.0)],                                # 4: X90, Y90
    [(90.0, 0.0), (90.0, 270.0)],                               # 5: X90, -Y90
    [(90.0, 180.0), (90.0, 90.0)],                              # 6: -X90, Y90
    [(90.0, 180.0), (90.0, 270.0)],                             # 7: -X90, -Y90
    [(90.0, 90.0), (90.0, 0.0)],                                # 8: Y90, X90
    [(90.0, 90.0), (90.0, 180.0)],                              # 9: Y90, -X90
    [(90.0, 270.0), (90.0, 0.0)],                               # 10: -Y90, X90
    [(90.0, 270.0), (90.0, 180.0)],                             # 11: -Y90, -X90
    [(90.0, 0.0)],                                              # 12: X90
    [(90.0, 180.0)],                                            # 13: -X90
    [(90.0, 90.0)],                                             # 14: Y90
    [(90.0, 270.0)],                                            # 15: -Y90
    [(90.0, 180.0), (90.0, 90.0), (90.0, 0.0)],                 # 16: -X90, Y90, X90
    [(90.0, 180.0), (90.0, 270.0), (90.0, 0.0)],                # 17: -X90, -Y90, X90
    [(180.0, 0.0), (90.0, 90.0)],                               # 18: X180, Y90
    [(180.0, 0.0), (90.0, 270.0)],                              # 19: X180, -Y90
    [(180.0, 90.0), (90.0, 0.0)],                               # 20: Y180, X90
    [(180.0, 90.0), (90.0, 180.0)],                             # 21: Y180, -X90
    [(90.0, 0.0), (90.0, 90.0), (90.0, 0.0)],                   # 22: X90, Y90, X90
    [(90.0, 180.0), (90.0, 90.0), (90.0, 180.0)],               # 23: -X90, Y90, -X90
]

_EYE = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _rotation_matrix(theta_deg: float, phi_deg: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    axis = np.cos(ph) * _SX + np.sin(ph) * _SY
    return np.cos(th / 2.0) * _EYE - 1j * np.sin(th / 2.0) * axis


def _build_clifford_unitaries() -> list[np.ndarray]:
    unitaries = []
    for pulse_list in CLIFFORD_PULSES:
        u = _EYE.copy()
        for theta, phi in pulse_list:
            u = _rotation_matrix(theta, phi) @ u
        unitaries.append(u)
    return unitaries


CLIFFORD_UNITARIES = _build_clifford_unitaries()


def find_recovery_gate(u_total: np.ndarray) -> int:
    """Find the Clifford gate index k whose unitary inverts ``u_total``.

    The group is closed, so the winning overlap is 1 up to float noise. Assert
    it: ``argmax`` alone always returns SOMETHING, so a typo in CLIFFORD_PULSES
    would silently append a wrong recovery gate and show up only as an
    RB curve that decays to the wrong asymptote — indistinguishable from a
    genuinely worse gate.
    """
    traces = [np.abs(np.trace(u_k @ u_total)) / 2.0 for u_k in CLIFFORD_UNITARIES]
    best = int(np.argmax(traces))
    if not np.isclose(traces[best], 1.0, atol=1e-6):
        raise RuntimeError(
            f"no Clifford inverts the accumulated sequence (best overlap "
            f"{traces[best]:.6f}, expected 1.0) — CLIFFORD_PULSES is not a "
            f"faithful decomposition of the 24-element group")
    return best


@register
class QbloxQubitSQRB(QubitSQRB):
    """Build a multiplexed Single Qubit Randomized Benchmarking Schedule for Qblox."""

    #: true of THIS probe only; scqo shows them in `scqo run qubit_sqrb --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the random sequences are drawn on the host and compiled into the schedule; with "
        "seed unset the generator is seeded with 42",
        "the identity Clifford plays nothing, so a Clifford averages 44 / 24 = 1.83 "
        "pulses here, while the analysis divides the error per Clifford by 1.875",
        "the pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange

        depths = [int(d) for d in self.sweep_axes["depth"]]
        num_sequences = int(self.params.num_random_sequences)
        reps = int(self.params.num_averages)
        max_depth = max(depths)
        seed = getattr(self.params, "seed", None)
        rng = np.random.default_rng(seed if seed is not None else 42)

        # Pre-generate random sequences for each sequence index
        sequences: list[list[int]] = []
        for _ in range(num_sequences):
            seq = [int(x) for x in rng.integers(0, 24, size=max_depth)]
            sequences.append(seq)

        schedule = Schedule("qubit_sqrb_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)
            sub = Schedule(f"qubit_sqrb_{qubit_name}")

            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                for m, seq in enumerate(sequences):
                    for d in depths:
                        add_reset(sub, self, qubit_name)

                        # Accumulate unitary and play gates for depth d
                        u_total = _EYE.copy()
                        for gate_idx in seq[:d]:
                            for theta, phi in CLIFFORD_PULSES[gate_idx]:
                                sub.add(Rxy(theta=theta, phi=phi, qubit=qubit_name))
                            u_total = CLIFFORD_UNITARIES[gate_idx] @ u_total

                        # Find and play recovery gate
                        rec_gate = find_recovery_gate(u_total)
                        for theta, phi in CLIFFORD_PULSES[rec_gate]:
                            sub.add(Rxy(theta=theta, phi=phi, qubit=qubit_name))

                        sub.add(
                            Measure(
                                qubit_name,
                                coords={
                                    f"seq_{qubit_name}": m,
                                    f"depth_{qubit_name}": d,
                                },
                                acq_channel=f"S_21_{qubit_name}",
                                **acq,
                            )
                        )
                        sub.add(IdlePulse(4e-9))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)

        return schedule
