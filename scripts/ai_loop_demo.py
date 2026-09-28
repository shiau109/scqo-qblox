"""Sketch of the loop an AI agent would drive over the SAME scqo.Session API.

The agent only ever sees JSON: a catalog of measurements (with parameter schemas), the
structured result of each run (now carrying ``run_id``/``data_path``), the queryable
run store (``find_runs``), and the device snapshot + change history it uses as memory.
No Qblox or QM object crosses the agent boundary — the identical loop works on either
backend.

    python scripts/ai_loop_demo.py
"""

from __future__ import annotations

import json

from scqo.cli import build_session, default_targets


def agent_decide(catalog: list[dict], device_state: dict,
                 targets: list[str]) -> tuple[str, dict]:
    """Stand-in for an LLM: pick an experiment + fill its parameters from the schema.

    A real agent would receive `catalog` as tool definitions and `device_state` as
    context, then emit a tool call. Here we hard-code one decision.

    ``targets`` are ENTITY names of the kind the experiment measures (qubit-like
    modes here) — the device state is keyed by every entity that carries knobs,
    channels and lines included (``fl.q0``, ``xy_q0.q0``), so it is context, not a
    target list.
    """
    _ = catalog, device_state
    return "resonator_spectroscopy", {"targets": targets,
                                      "start_readout_detuning_hz": -7.5e6,
                                      "end_readout_detuning_hz": 7.5e6}


def main() -> None:
    sess, _ = build_session()

    # 1. perceive: what can I measure, and what is the device's current state?
    catalog = sess.catalog()
    print("catalog:", json.dumps([c["name"] for c in catalog], indent=2))

    # 2. decide -> 3. act -> 4. read result -> (loop)
    for _ in range(1):  # one iteration for the demo; a real loop continues until a goal is met
        # default_targets = every roster entity of the kind a single-qubit
        # experiment measures (qubit-like modes), i.e. the agent's menu of targets
        experiment, params = agent_decide(
            catalog, sess.device_state(), default_targets(sess))
        # update="apply": an unattended agent applies its own results (scqo v0.6.0);
        # the default "suggest" would leave the state below unchanged and history empty.
        result = sess.run(experiment, params, update="apply")
        print(f"ran {experiment} -> success={all(v == 'successful' for v in result['outcomes'].values())}")
        print("extracted:", json.dumps(result["fit"], indent=2))
        if "run_id" in result:  # the loop's episodic memory: every run is findable later
            print("saved as:", result["run_id"])

    # 5. remember: past runs + change history are the loop's memory
    print("recent runs:", json.dumps([r["run_id"] for r in sess.find_runs(limit=5)], indent=2))
    print("final device state:", json.dumps(sess.device_state(), indent=2))


if __name__ == "__main__":
    main()
