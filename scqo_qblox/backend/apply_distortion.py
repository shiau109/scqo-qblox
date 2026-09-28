"""One command: accepted cryoscope distortion facts -> the Qblox predistortion config.

After ``scqo run qubit_spectroscopy_cryoscope --target q1`` (or the ramsey one) +
``scqo accept``, the fit's ``distortion_amp``/``distortion_tau_s`` live as FACTS in
scqo's ``physical.json`` — record-only, never auto-pushed. This turns them into the
QCM's real-time predistortion filter in one step: resolve the ACTIVE scqo
device/setup (the same selection ``scqo run`` uses), read the taps, convert them
through :func:`scqo_qblox.backend._distortion.to_qblox_distortion` (the 4-stage
exponential-overshoot bank), key the ONE resulting
``QbloxHardwareDistortionCorrection`` into
``hardware_options.distortion_corrections["<flux_port>-cl0.baseband"]``
(e.g. ``"q1:fl-cl0.baseband"`` — flux plays on the baseband identity clock), and
save both config files (``QbloxDeviceModel.save``). Fully OFFLINE — no cluster is
contacted; the correction is compiled and pushed on the next run that plays flux.

Run it (in ``.venv-qblox``)::

    python -m scqo_qblox.backend.apply_distortion --target q1
    python -m scqo_qblox.backend.apply_distortion --target q1 --dry-run   # preview only
    python -m scqo_qblox.backend.apply_distortion --target q1 --extend    # merge a residual

It never runs automatically on ``scqo accept`` — applying predistortion is a
deliberate, opt-in step (measure a fresh full correction on a filter-CLEARED line).

Two sources for the taps, exactly like the QM sibling
(``scqo_qm.backend.apply_distortion``):

* ``--run <run_id>`` reads the fit straight from that run's ``result.json`` —
  the iteration door: it names exactly which measurement feeds the filter, so
  ``scqo accept`` order is irrelevant to the vendor config (the two cryoscopes
  share ONE fact slot with REPLACE semantics).
* No ``--run``: the accepted facts (``physical.json``).

THE 4-STAGE BANK IS A HARD HARDWARE LIMIT, which shapes two behaviors the QM
sibling does not have: (1) taps beyond the bank (or out of the per-stage bounds:
|A| <= 1, tau >= 6 ns) land in ``overflow`` and are reported LOUDLY — kept stages
are the most significant by |A|, and all-overflow refuses; (2) ``--extend`` cannot
append — it merges the existing stages with the new taps and re-partitions the
whole set, so a residual refinement re-ranks against what is already applied.

Facts vs filter diverge by design (same rationale as the QM CLI): the facts hold
the latest accepted MEASUREMENT, the vendor entry the accumulated applied
CORRECTION. Never write a composed total back into the facts — ``scqo accept``'s
staleness guard would strand every pending cryoscope suggestion behind ``--force``.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

from scqo_qblox.backend._distortion import to_qblox_distortion

#: the roster channel kind of a qubit's flux line (catalog CHANNELS).
FLUX_KIND = "flux"

#: experiments whose result.fit carries the distortion taps.
CRYOSCOPE_EXPERIMENTS = ("qubit_spectroscopy_cryoscope", "qubit_ramsey_cryoscope")

#: flux operations play on the baseband identity clock
#: (``BasebandClockResource.IDENTITY``) — the port-clock key's second half.
BASEBAND_CLOCK = "cl0.baseband"

#: the model's four per-stage slots, in order.
_STAGE_KEYS = ("exp0_coeffs", "exp1_coeffs", "exp2_coeffs", "exp3_coeffs")


def _run_taps(session: Any, run_id: str, target: str) -> tuple[list, list]:
    """The fitted ``(amps, taus_s)`` from one saved run — refuses BY NAME a
    non-cryoscope run, a failed target, or a run without the fit fields."""
    try:
        data = session.load_run(run_id)
    except KeyError as err:
        raise SystemExit(err.args[0] if err.args else str(err)) from None
    record = data["record"]
    experiment = record.get("experiment")
    if experiment not in CRYOSCOPE_EXPERIMENTS:
        raise SystemExit(
            f"run {run_id} is a {experiment!r} run — distortion taps come from "
            f"one of {', '.join(CRYOSCOPE_EXPERIMENTS)}"
        )
    outcome = (record.get("outcomes") or {}).get(target)
    if outcome != "successful":
        raise SystemExit(
            f"run {run_id}: {target!r} outcome is {outcome!r}, not 'successful' "
            f"— refusing to apply a failed fit"
        )
    fit = (data.get("result", {}).get("fit") or {}).get(target) or {}
    amps = fit.get("distortion_amp")
    taus_s = fit.get("distortion_tau_s")
    if amps is None or taus_s is None:
        raise SystemExit(
            f"run {run_id}: no distortion_amp/distortion_tau_s in result.fit "
            f"for {target!r}"
        )
    return amps, taus_s


def _flux_portclock(session: Any, target: str) -> tuple[str, str]:
    """``(line, port-clock key)`` for ``target``'s flux line — the roster
    resolves the line (q1 -> z1, which owns the taps since 4.0.0), the vendor
    element behind it names the port."""
    roster = session.backend.roster
    line = roster.entities[roster.default_channel(target, FLUX_KIND)].line
    port = session.backend.device.component(line)._element.ports.flux
    return line, f"{port}-{BASEBAND_CLOCK}"


def _hardware_options(session: Any) -> Any:
    """The validated hardware options this setup compiles against — the same
    authoritative object the output-attenuation writes go through."""
    return session.backend._hw_agent.hardware_configuration.hardware_options


def _stage_value(stage: Any, field: str) -> float:
    return float(stage[field] if isinstance(stage, dict) else getattr(stage, field))


def _existing_pairs(entry: Any, portclock: str) -> list[tuple[float, float]]:
    """The ``(amplitude, tau_s)`` pairs already applied under ``portclock`` —
    reads a validated model or a raw dict alike; a LIST entry is refused loudly
    (a real/flux output takes exactly ONE correction — a list means the key was
    written for a complex channel, which this line is not)."""
    if entry is None:
        return []
    if isinstance(entry, list):
        raise SystemExit(
            f"distortion_corrections[{portclock!r}] holds a LIST of corrections "
            f"— that is the complex-channel shape, and a flux (real) output "
            f"takes exactly one. Fix the config by hand before applying."
        )
    pairs: list[tuple[float, float]] = []
    for key in _STAGE_KEYS:
        stage = entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
        if stage is None:
            continue
        pairs.append((_stage_value(stage, "amplitude"),
                      _stage_value(stage, "time_constant")))
    return pairs


def _corrections_dict(opts: Any) -> dict:
    """The mutable ``distortion_corrections`` mapping, created when absent."""
    if opts.distortion_corrections is None:
        opts.distortion_corrections = {}
    return opts.distortion_corrections


def _config_paths(session: Any) -> list[str]:
    device = session.backend.device
    return [str(p) for p in (getattr(device, "_hw_config_file", None),
                             getattr(device, "_config_file", None)) if p]


def clear_distortion(
    target: str,
    *,
    config_path: str | None = None,
    session: Any = None,
    save: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove ``target``'s distortion-correction entry — the fresh-line reset
    before a clean-slate cryoscope characterization. Returns
    ``{"target", "line", "portclock", "removed", "config_files", "saved"}``.
    OFFLINE.
    """
    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, _cfg = build_session(config_path)
    line, portclock = _flux_portclock(session, target)
    opts = _hardware_options(session)
    corrections = opts.distortion_corrections or {}
    removed = _existing_pairs(corrections.get(portclock), portclock)
    if not dry_run and portclock in corrections:
        del corrections[portclock]
    did_save = bool(save and not dry_run)
    if did_save:
        session.backend.device.save()
    return {"target": target, "line": line, "portclock": portclock,
            "removed": removed, "config_files": _config_paths(session),
            "saved": did_save}


def apply_distortion_from_state(
    target: str,
    *,
    run_id: str | None = None,
    replace: bool = True,
    config_path: str | None = None,
    session: Any = None,
    save: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply distortion taps for ``target`` to the Qblox predistortion config.

    Resolves the ACTIVE scqo selection (unless ``session`` is injected — for
    tests) and takes the taps from ``run_id``'s saved fit when given, else from
    the accepted facts on the target's flux LINE. Converts them through the
    4-stage bank (:func:`to_qblox_distortion`; overflow reported LOUDLY, never
    silently dropped), writes ONE ``QbloxHardwareDistortionCorrection`` under
    ``"<flux_port>-cl0.baseband"`` and (unless ``dry_run``/``save=False``) saves
    both config files. ``replace=False`` merges the existing stages with the new
    taps and re-partitions — the bank cannot append. OFFLINE.

    Returns a summary dict: ``target``, ``line``, ``portclock``, ``run_id``,
    ``amps``, ``taus_s``, ``existing_taps``, ``kept``, ``overflow``,
    ``config_files``, ``saved``. Raises ``SystemExit`` when no taps are
    available or none is representable.
    """
    if session is None:
        from scqo.cli import build_session  # lazy: keep module import scqo-free

        session, _cfg = build_session(config_path)

    line, portclock = _flux_portclock(session, target)
    if run_id is not None:
        amps, taus_s = _run_taps(session, run_id, target)
    else:
        amps = session.physical.get(line, "distortion_amp")
        taus_s = session.physical.get(line, "distortion_tau_s")
        if amps is None or taus_s is None:
            raise SystemExit(
                f"no accepted distortion facts on {line} — run and accept a "
                f"cryoscope for {target!r} first (distortion_amp/distortion_tau_s "
                f"are unset in physical.json), or apply straight from a run with "
                f"--run <run_id>"
            )

    opts = _hardware_options(session)
    corrections = _corrections_dict(opts)
    existing = _existing_pairs(corrections.get(portclock), portclock)

    pairs = list(zip([float(a) for a in amps], [float(t) for t in taus_s]))
    if replace and existing:
        warnings.warn(
            f"replacing {len(existing)} existing distortion stage(s) on "
            f"{portclock}; a full correction must be MEASURED on a "
            f"filter-cleared line (use --extend to refine a residual instead)",
            stacklevel=2,
        )
    if not replace:
        pairs = existing + pairs

    block = to_qblox_distortion([a for a, _ in pairs], [t for _, t in pairs])
    overflow = block.pop("overflow")
    if overflow:
        dropped = ", ".join(f"A={a:+.5g} tau={tau * 1e9:.4g} ns"
                            for a, tau in overflow)
        warnings.warn(
            f"{portclock}: {len(overflow)} tap(s) exceed the QCM's 4-stage "
            f"exponential bank (or its |A|<=1 / tau>=6 ns bounds) and are NOT "
            f"applied: {dropped}. Kept the most significant by |A|; the "
            f"wideband/FIR path is deferred.",
            stacklevel=2,
        )
    if not block:
        raise SystemExit(
            f"{portclock}: none of the {len(pairs)} tap(s) is representable in "
            f"the QCM's exponential bank (|A| <= 1, tau >= 6 ns) — nothing to "
            f"apply."
        )

    from qblox_scheduler.backends.types.qblox import QbloxHardwareDistortionCorrection

    corrections[portclock] = QbloxHardwareDistortionCorrection.model_validate(block)
    kept = _existing_pairs(corrections[portclock], portclock)

    did_save = bool(save and not dry_run)
    if did_save:
        session.backend.device.save()

    return {
        "target": target,
        "line": line,
        "portclock": portclock,
        "run_id": run_id,
        "amps": [float(a) for a in amps],
        "taus_s": [float(t) for t in taus_s],
        "existing_taps": len(existing),
        "kept": kept,
        "overflow": [tuple(p) for p in overflow],
        "config_files": _config_paths(session),
        "saved": did_save,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scqo_qblox.backend.apply_distortion",
        description="Apply accepted cryoscope distortion taps to the Qblox "
        "hardware distortion-correction config for the ACTIVE scqo device/setup.",
    )
    p.add_argument("--target", required=True, help="qubit/mode name, e.g. q1")
    p.add_argument(
        "--run",
        default=None,
        metavar="RUN_ID",
        help="take the taps from this saved run's fit instead of the accepted "
        "facts (the iteration door; accept order becomes irrelevant)",
    )
    p.add_argument(
        "--extend",
        action="store_true",
        help="merge the new taps with the already-applied stages and "
        "re-partition the 4-stage bank (refine a residual) instead of "
        "overwriting it",
    )
    p.add_argument(
        "--config", default=None, help="scqo config.toml path (default: active selection)"
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="remove the target's distortion-correction entry (the fresh-line "
        "reset before a clean-slate characterization)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve + preview what would be written; save nothing",
    )
    args = p.parse_args(argv)

    if args.clear:
        if args.run or args.extend:
            p.error("--clear takes no --run/--extend")
        out = clear_distortion(args.target, config_path=args.config,
                               dry_run=args.dry_run)
        verb = "would remove" if args.dry_run else "removed"
        print(f"{args.target} ({out['portclock']}): {verb} "
              f"{len(out['removed'])} distortion stage(s)")
        for a, tau_s in out["removed"]:
            print(f"    A={a:+.5g}  tau={tau_s * 1e9:.4g} ns")
        if args.dry_run:
            print("  --dry-run: nothing written")
        else:
            for path in out["config_files"]:
                print(f"  saved: {path}")
        return 0

    out = apply_distortion_from_state(
        args.target,
        run_id=args.run,
        replace=not args.extend,
        config_path=args.config,
        dry_run=args.dry_run,
    )

    verb = "would write" if args.dry_run else ("merged into" if args.extend else "wrote")
    source = f"run {out['run_id']}" if out["run_id"] else "accepted facts"
    print(
        f"{args.target} ({out['portclock']}, from {source}): {verb} "
        f"{len(out['kept'])} distortion stage(s)"
    )
    for a, tau_s in out["kept"]:
        print(f"    A={a:+.5g}  tau={tau_s * 1e9:.4g} ns")
    for a, tau_s in out["overflow"]:
        print(f"    DROPPED (bank full/bounds): A={a:+.5g}  tau={tau_s * 1e9:.4g} ns")
    if args.dry_run:
        print("  --dry-run: nothing written")
    else:
        for path in out["config_files"]:
            print(f"  saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
