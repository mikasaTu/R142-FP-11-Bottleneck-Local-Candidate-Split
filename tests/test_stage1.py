from __future__ import annotations

import json

from r142_bottleneck.benchmark import BenchmarkConfig, ForkPush2D
from r142_bottleneck.detector import detect_earliest_bottleneck
from r142_bottleneck.experiment import ExperimentConfig, aggregate_results, load_episode_metrics, run_experiment
from r142_bottleneck.methods import POLICIES, generate_policy_set


def center_episode(env: ForkPush2D):
    for index in range(100):
        spec = env.make_episode(index, 10_000 + index)
        if abs(spec.shared_selector) < 0.2:
            return spec
    raise AssertionError("could not construct center-collapse episode")


def test_two_symmetric_success_modes_and_central_failure():
    env = ForkPush2D(BenchmarkConfig())
    spec = center_episode(env)
    step = spec.true_bottleneck_step
    upper = env.rollout(spec, 1, {step: 1.0})
    lower = env.rollout(spec, 2, {step: -1.0})
    center = env.rollout(spec, 3)
    assert upper["success"] and upper["mode"] == "upper"
    assert lower["success"] and lower["mode"] == "lower"
    assert not center["success"]


def test_detector_localizes_without_terminal_inputs():
    env = ForkPush2D(BenchmarkConfig())
    spec = center_episode(env)
    scouts = generate_policy_set(env, spec, "B0_best_of_n").candidates[: env.config.scout_count]
    prediction, diagnostics = detect_earliest_bottleneck(scouts, env.config.horizon)
    assert prediction == spec.true_bottleneck_step
    assert diagnostics["terminal_label_access"] is False


def test_all_policies_preserve_budget_except_declared_more_samples():
    env = ForkPush2D(BenchmarkConfig())
    spec = center_episode(env)
    for policy in POLICIES:
        candidate_set = generate_policy_set(env, spec, policy)
        expected = env.config.more_samples_budget if policy == "A_more_samples" else env.config.candidate_budget
        assert len(candidate_set.candidates) == expected
    proposed = generate_policy_set(env, spec, "proposed_bottleneck_local")
    children = proposed.candidates[env.config.scout_count :]
    assert all(child.parent_id is not None for child in children)
    assert all(child.split_action_step == spec.true_bottleneck_step for child in children)


def test_wrong_location_does_not_change_center_route():
    env = ForkPush2D(BenchmarkConfig())
    spec = center_episode(env)
    proposed = generate_policy_set(env, spec, "proposed_bottleneck_local")
    wrong = generate_policy_set(env, spec, "A_wrong_late")
    assert proposed.selected().final_success
    assert not wrong.selected().final_success


def test_shard_writes_complete_genealogy_and_aggregate(tmp_path):
    bench = BenchmarkConfig()
    exp = ExperimentConfig(evaluation_seeds=6, bootstrap_replicates=100, block_size=1, minimum_winning_blocks=1)
    shard = run_experiment(tmp_path / "shard", bench, exp, save_genealogy=True)
    assert (shard / "COMPLETE").read_text().strip() == "complete"
    manifest = json.loads((shard / "shard_manifest.json").read_text())
    assert manifest["complete"] is True
    records = load_episode_metrics([shard / "episode_metrics.jsonl"])
    assert len(records) == 6 * len(POLICIES)
    result = aggregate_results(records, tmp_path / "aggregate", exp)
    assert "decision" in result["gate"]
    assert (tmp_path / "aggregate" / "mechanism_diagnostics.json").exists()
