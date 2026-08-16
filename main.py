"""Command-line entrypoint for Cloud-Orchestra.

Subcommands:
  run            Run a single remediation workflow (default: the high_cpu demo).
  eval           Run the evaluation harness and print the four KPIs.
  ablation       Run the five ablation studies and print the comparison table.
  train-finops   Train the PPO FinOps policy (requires torch).
  api            Start the FastAPI control plane.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from cloud_orchestra.core.config import get_settings
from cloud_orchestra.core.logging import configure_logging
from cloud_orchestra.schemas import Alert


def _demo_alert() -> Alert:
    from cloud_orchestra.eval.scenarios import build_scenarios

    scenarios = {s.id: s for s in build_scenarios()}
    return scenarios["high_cpu"].alert


async def _run(args: argparse.Namespace) -> None:
    from cloud_orchestra.eval.scenarios import build_scenarios
    from cloud_orchestra.runtime import Runtime

    settings = get_settings()
    runtime = Runtime(settings, persistent=True)
    await runtime.init_db()
    try:
        scenarios = {s.id: s for s in build_scenarios()}
        alert = scenarios[args.problem].alert if args.problem in scenarios else _demo_alert()
        result = await runtime.run_alert(alert)
        print(json.dumps({
            "run_id": str(result.run.id),
            "status": result.run.status.value,
            "provider": result.run.provider.value if result.run.provider else None,
            "resolved": result.run.resolved,
            "cost_before": result.run.cost_before,
            "cost_after": result.run.cost_after,
            "findings": len(result.findings),
            "elapsed_ms": round(result.elapsed_ms, 1),
        }, indent=2))
        if result.explanation:
            print("\n" + result.explanation)
    finally:
        await runtime.close()


async def _eval() -> None:
    from cloud_orchestra.eval.harness import EvaluationHarness
    from cloud_orchestra.eval.metrics import metrics_table

    harness = EvaluationHarness()
    _results, metrics = await harness.run()
    print(metrics_table([("baseline", metrics)]))
    print(json.dumps(metrics.to_dict(), indent=2))


async def _ablation() -> None:
    from cloud_orchestra.eval.ablations import compute_deltas, format_report
    from cloud_orchestra.eval.harness import EvaluationHarness

    harness = EvaluationHarness()
    results = await harness.run_all_ablations()
    print(format_report(results))
    print("\nDeltas vs baseline:")
    print(json.dumps(compute_deltas(results), indent=2))


def _train_finops(args: argparse.Namespace) -> None:
    from cloud_orchestra.rl.env import FinOpsEnv

    try:
        from cloud_orchestra.rl.ppo import PPOTrainer
    except ImportError as exc:
        print("PyTorch is required for RL training. Install with `pip install -e '.[rl]'`.")
        raise SystemExit(1) from exc

    env = FinOpsEnv(demand=2, horizon=8)
    trainer = PPOTrainer()
    history = trainer.train(env, iterations=args.iterations)
    if args.output:
        trainer.save(args.output)
    print(f"trained {args.iterations} iterations; final episode reward = {history[-1]:.2f}")
    print(f"reward history: {[round(r, 2) for r in history]}")


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="cloud-orchestra")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a single remediation workflow")
    run_p.add_argument("--problem", default="high_cpu", help="scenario/problem class to run")

    sub.add_parser("eval", help="run the evaluation harness")

    sub.add_parser("ablation", help="run the five ablation studies")

    train_p = sub.add_parser("train-finops", help="train the PPO FinOps policy")
    train_p.add_argument("--iterations", type=int, default=60)
    train_p.add_argument("--output", default="./models/finops_ppo.pt")

    sub.add_parser("api", help="start the FastAPI control plane")

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    if args.command == "api":
        import uvicorn

        uvicorn.run("cloud_orchestra.api.app:app", host="0.0.0.0", port=8000)
    elif args.command == "eval":
        asyncio.run(_eval())
    elif args.command == "ablation":
        asyncio.run(_ablation())
    elif args.command == "train-finops":
        _train_finops(args)
    else:
        asyncio.run(_run(args))


if __name__ == "__main__":
    main(sys.argv[1:])
