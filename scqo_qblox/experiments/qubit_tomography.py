"""Qblox Qubit Tomography acquisition probe.

Performs quantum state tomography by applying initial state preparation,
sweeping target gate repetitions, applying basis rotations, and recording
single-shot training and tomography data.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitTomography``.
"""

from __future__ import annotations

from typing import Any, ClassVar
import numpy as np
import xarray as xr

from scqo import register
from scqo.experiments import QubitTomography

from ._reset import add_reset
from ._state import measure_kwargs


def _play_init_state(sub: Any, q_name: str, state_str: str) -> None:
    """Add initial state preparation pulses to sub-schedule."""
    from qblox_scheduler.operations import Rxy

    st = str(state_str).strip().lower()
    if st in ("0", "g"):
        pass
    elif st in ("1", "e"):
        sub.add(Rxy(theta=180.0, phi=0.0, qubit=q_name))
    elif st in ("+", "+x"):
        sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
    elif st in ("-", "-x"):
        sub.add(Rxy(theta=90.0, phi=270.0, qubit=q_name))
    elif st in ("+i", "+y"):
        sub.add(Rxy(theta=90.0, phi=180.0, qubit=q_name))
    elif st in ("-i", "-y"):
        sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
    else:
        raise ValueError(
            f"Unsupported initial state '{state_str}'. "
            "Supported states are: '0', 'g', '1', 'e', '+', '+x', '-', '-x', '+i', '+y', '-i', '-y'."
        )


def _play_target_gate(sub: Any, q_name: str, gate_str: str) -> None:
    """Add target gate pulse to sub-schedule."""
    from qblox_scheduler.operations import Rxy

    gt = str(gate_str).strip().lower()
    if gt in ("i", "id"):
        pass
    elif gt in ("x", "x180"):
        sub.add(Rxy(theta=180.0, phi=0.0, qubit=q_name))
    elif gt in ("x90", "x/2"):
        sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))
    elif gt in ("y", "y180"):
        sub.add(Rxy(theta=180.0, phi=90.0, qubit=q_name))
    elif gt in ("y90", "y/2"):
        sub.add(Rxy(theta=90.0, phi=90.0, qubit=q_name))
    elif gt in ("-x90", "-x/2"):
        sub.add(Rxy(theta=90.0, phi=180.0, qubit=q_name))
    elif gt in ("-y90", "-y/2"):
        sub.add(Rxy(theta=90.0, phi=270.0, qubit=q_name))
    else:
        raise ValueError(
            f"Unsupported target gate '{gate_str}'. "
            "Supported gates are: 'i', 'id', 'x', 'x180', 'x90', 'x/2', 'y', 'y180', 'y90', 'y/2', '-x90', '-y90'."
        )


def _play_basis_rotation(sub: Any, q_name: str, basis_str: str) -> None:
    """Add basis measurement rotation pulse to sub-schedule."""
    from qblox_scheduler.operations import Rxy

    b = str(basis_str).strip().lower()
    if b == "z":
        pass
    elif b == "x":
        sub.add(Rxy(theta=90.0, phi=270.0, qubit=q_name))  # -Y90
    elif b == "y":
        sub.add(Rxy(theta=90.0, phi=0.0, qubit=q_name))    # X90
    else:
        raise ValueError(
            f"Unsupported measurement basis '{basis_str}'. Supported bases are: 'z', 'x', 'y'."
        )


@register
class QbloxQubitTomography(QubitTomography):
    """Build and execute state tomography with GMM training and multi-basis readout on Qblox."""

    #: true of THIS probe only; scqo shows them in `scqo run qubit_tomography --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the amp and detuning entries of qubit_configs are not realized: every gate is "
        "played as calibrated",
        "the pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
        "--preview is refused: the probe acquires the training and the tomography shots "
        "itself",
    )

    probe_self_acquires = (
        "it acquires GMM training shots and state tomography shots across schedules"
    )

    def plan(self) -> dict:
        """Every axis this probe will sweep, resolved once.

        Both schedule builders and the decode need the same lists, and the
        noise axis in particular is DERIVED (a spectator qubit plus
        ``interleave_noise``), so recomputing it in three places is how the
        dataset and the program drift apart.
        """
        qubits = list(self.params.targets)
        qubit_configs = self.params.qubit_configs
        has_noise = any(
            bool(qubit_configs.get(q, {}).get("noise_mode", False))
            for q in qubits
        )
        interleave = getattr(self.params, "interleave_noise", True)
        return {
            "qubits": qubits,
            "qubit_configs": qubit_configs,
            "gate_counts": [int(gc) for gc in self.params.gate_counts],
            "num_shots": int(self.params.num_averages),
            "num_training_shots": int(self.params.num_training_shots),
            "noise_conditions": ["off", "on"] if (has_noise and interleave) else ["off"],
            "bases": ["z", "x", "y"],
            "sym_names": ["reg", "inv"] if self.params.symmetrized_readout else ["reg"],
        }

    def build_training_schedule(self, plan: dict | None = None) -> Any:
        """The |g>/|e> block that trains the discriminator.

        Split out of probe() so tests/test_probe_surface.py can COMPILE it: a
        self-acquiring probe otherwise gets no compile coverage, and the time
        grid, the DAC range and the latched-parameter alignment all live in the
        compiler rather than in the built tree.
        """
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange

        plan = plan or self.plan()
        qubits = plan["qubits"]
        qubit_configs = plan["qubit_configs"]
        num_training_shots = plan["num_training_shots"]

        sched_tr = Schedule("tomography_training_multiplexed")
        for q_name in qubits:
            noise_mode = bool(qubit_configs.get(q_name, {}).get("noise_mode", False))
            acq = measure_kwargs(self, q_name)
            sub_tr = Schedule(f"tomography_tr_{q_name}")

            # Prepared state 0: Reset -> Measure
            with sub_tr.loop(arange(0, num_training_shots, 1, DType.NUMBER)) as shot:
                add_reset(sub_tr, self, q_name)
                sub_tr.add(
                    Measure(
                        q_name,
                        coords={f"state_{q_name}": 0, f"shot_{q_name}": shot},
                        acq_channel=f"S_21_{q_name}",
                        **acq,
                    )
                )
                sub_tr.add(IdlePulse(4e-9))

            # Prepared state 1: Reset -> X180 -> Measure
            with sub_tr.loop(arange(0, num_training_shots, 1, DType.NUMBER)) as shot:
                add_reset(sub_tr, self, q_name)
                if not noise_mode:
                    sub_tr.add(Rxy(theta=180.0, phi=0.0, qubit=q_name))
                sub_tr.add(
                    Measure(
                        q_name,
                        coords={f"state_{q_name}": 1, f"shot_{q_name}": shot},
                        acq_channel=f"S_21_{q_name}",
                        **acq,
                    )
                )
                sub_tr.add(IdlePulse(4e-9))

            sub_tr.add(IdlePulse(4e-9))
            sched_tr.add(sub_tr)
        return sched_tr

    def build_tomography_schedule(self, plan: dict | None = None) -> Any:
        """The basis x symmetrisation x gate-count block. See
        :meth:`build_training_schedule` for why it is a separate method."""
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy
        from qblox_scheduler.operations.loop_domains import DType, arange

        plan = plan or self.plan()
        qubits = plan["qubits"]
        qubit_configs = plan["qubit_configs"]
        gate_counts = plan["gate_counts"]
        num_shots = plan["num_shots"]
        noise_conditions = plan["noise_conditions"]
        bases = plan["bases"]
        sym_names = plan["sym_names"]

        sched_tomo = Schedule("tomography_multiplexed")
        for q_name in qubits:
            q_cfg = qubit_configs.get(q_name, {})
            noise_mode = bool(q_cfg.get("noise_mode", False))
            init_st = q_cfg.get("init_state", "0")
            tgt_gt = q_cfg.get("target_gate", "X180")
            acq = measure_kwargs(self, q_name)
            sub_tomo = Schedule(f"tomography_{q_name}")

            with sub_tomo.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                for nc_idx, nc in enumerate(noise_conditions):
                    for b_idx, basis in enumerate(bases):
                        for s_idx, sym in enumerate(sym_names):
                            for gc_idx, gc in enumerate(gate_counts):
                                add_reset(sub_tomo, self, q_name)

                                if not noise_mode:
                                    # Phase 1: Init State
                                    _play_init_state(sub_tomo, q_name, init_st)

                                    # Phase 2: Target Gate repeated gc times
                                    for _ in range(gc):
                                        _play_target_gate(sub_tomo, q_name, tgt_gt)

                                    # Phase 3: Basis Rotation + Symmetrization
                                    _play_basis_rotation(sub_tomo, q_name, basis)
                                    if sym == "inv":
                                        sub_tomo.add(Rxy(theta=180.0, phi=0.0, qubit=q_name))
                                else:
                                    # Spectator noise mode: only inject noise when active
                                    if nc == "on" or len(noise_conditions) == 1:
                                        for _ in range(gc):
                                            _play_target_gate(sub_tomo, q_name, tgt_gt)

                                sub_tomo.add(
                                    Measure(
                                        q_name,
                                        coords={
                                            f"nc_{q_name}": nc_idx,
                                            f"b_{q_name}": b_idx,
                                            f"s_{q_name}": s_idx,
                                            f"gc_{q_name}": gc_idx,
                                            f"shot_{q_name}": shot,
                                        },
                                        acq_channel=f"S_21_{q_name}",
                                        **acq,
                                    )
                                )
                                sub_tomo.add(IdlePulse(4e-9))
            sub_tomo.add(IdlePulse(4e-9))
            sched_tomo.add(sub_tomo)
        return sched_tomo

    def probe(self) -> Any:
        from scqo_qblox.backend.qblox_backend import chunk_timeout_s

        plan = self.plan()
        qubits = plan["qubits"]
        n_qubits = len(qubits)
        qubit_configs = plan["qubit_configs"]
        gate_counts = plan["gate_counts"]
        num_shots = plan["num_shots"]
        num_training_shots = plan["num_training_shots"]
        noise_conditions = plan["noise_conditions"]
        bases = plan["bases"]
        sym_names = plan["sym_names"]

        n_nc = len(noise_conditions)
        n_bases = len(bases)
        n_sym = len(sym_names)
        n_gc = len(gate_counts)

        hw_agent = self.backend._hw_agent

        # ====================================================================
        # 1. Training shots (|0> and |1> calibration)
        # ====================================================================
        i_train = np.zeros((n_qubits, 2, num_training_shots), dtype=float)
        q_train = np.zeros((n_qubits, 2, num_training_shots), dtype=float)

        sched_tr = self.build_training_schedule(plan)
        # the training block prepares |g> and |e>: two points per shot
        timeout_tr = chunk_timeout_s(self, shots=num_training_shots, points=2)
        hw_agent.instrument_coordinator.timeout(timeout_tr)
        raw_tr = hw_agent.run(sched_tr, timeout=timeout_tr)

        for k, q_name in enumerate(qubits):
            if bool(qubit_configs.get(q_name, {}).get("noise_mode", False)):
                continue
            key = f"S_21_{q_name}"
            if key not in raw_tr.data_vars:
                raise KeyError(f"acquisition channel {key!r} not in raw training dataset")
            tr_vals = np.asarray(raw_tr[key].values).reshape(2, num_training_shots)
            i_train[k] = np.real(tr_vals)
            q_train[k] = np.imag(tr_vals)

        # ====================================================================
        # 2. Tomography shots
        # ====================================================================
        i_tomo = np.zeros((n_qubits, n_nc, n_bases, n_sym, n_gc, num_shots), dtype=float)
        q_tomo = np.zeros((n_qubits, n_nc, n_bases, n_sym, n_gc, num_shots), dtype=float)

        sched_tomo = self.build_tomography_schedule(plan)
        total_tomo_points = num_shots * n_nc * n_bases * n_sym * n_gc
        timeout_tomo = chunk_timeout_s(self, shots=1, points=total_tomo_points)
        hw_agent.instrument_coordinator.timeout(timeout_tomo)
        raw_tomo = hw_agent.run(sched_tomo, timeout=timeout_tomo)

        for k, q_name in enumerate(qubits):
            noise_mode = bool(qubit_configs.get(q_name, {}).get("noise_mode", False))
            if noise_mode:
                continue
            key = f"S_21_{q_name}"
            if key not in raw_tomo.data_vars:
                raise KeyError(f"acquisition channel {key!r} not in raw tomography dataset")
            tomo_vals = np.asarray(raw_tomo[key].values).reshape(
                num_shots, n_nc, n_bases, n_sym, n_gc
            )
            # Transpose from (shot, nc, basis, sym, gc) to (nc, basis, sym, gc, shot)
            tomo_vals_transposed = np.transpose(tomo_vals, (1, 2, 3, 4, 0))
            i_tomo[k] = np.real(tomo_vals_transposed)
            q_tomo[k] = np.imag(tomo_vals_transposed)

        # =========================================================================
        # 3. Assemble and return Dataset conforming to TomographyContract
        # =========================================================================
        return xr.Dataset(
            data_vars={
                "I_tomo": (
                    ("target", "noise_condition", "basis", "sym", "gate_count", "shot_idx"),
                    i_tomo,
                ),
                "Q_tomo": (
                    ("target", "noise_condition", "basis", "sym", "gate_count", "shot_idx"),
                    q_tomo,
                ),
                "I_train": (("target", "prepared_state", "train_shot_idx"), i_train),
                "Q_train": (("target", "prepared_state", "train_shot_idx"), q_train),
            },
            coords={
                "target": qubits,
                "noise_condition": noise_conditions,
                "basis": bases,
                "sym": sym_names,
                "gate_count": np.array(gate_counts, dtype=int),
                "shot_idx": np.arange(num_shots, dtype=int),
                "prepared_state": np.array([0, 1], dtype=int),
                "train_shot_idx": np.arange(num_training_shots, dtype=int),
            },
        )
