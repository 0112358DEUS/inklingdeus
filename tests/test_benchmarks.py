#!/usr/bin/env python3
"""No-GPU regression tests for benchmark result handling."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))

import chat_bench  # noqa: E402
import compare_ab  # noqa: E402
import concurrency_bench  # noqa: E402
import evaluate_boot_cache  # noqa: E402
import evaluate_memfrac  # noqa: E402
import gsm8k_eval  # noqa: E402
import niah_eval  # noqa: E402
import repo_fingerprint  # noqa: E402
import select_block_size  # noqa: E402
import tool_call_regression  # noqa: E402


def response_payload(
    *,
    accept: float | None = 2.25,
    verify: int | None = None,
    histogram: list[int] | None = None,
):
    meta = {}
    if accept is not None:
        meta["spec_accept_length"] = accept
    if verify is not None:
        meta["spec_verify_ct"] = verify
    if histogram is not None:
        meta["spec_correct_drafts_histogram"] = histogram
    return {
        "usage": {"completion_tokens": 18},
        "choices": [{"message": {"content": "ok"}, "meta_info": meta}],
    }


class ChatBenchTests(unittest.TestCase):
    def run_sample(self, payload):
        response = io.BytesIO(json.dumps(payload).encode())
        with mock.patch.object(chat_bench.urllib.request, "urlopen", return_value=response):
            with mock.patch.object(
                chat_bench.time, "perf_counter", side_effect=(10.0, 12.0)
            ):
                return chat_bench.run_request(
                    url="http://example.invalid",
                    model="inkling-small",
                    prompt="test",
                    max_tokens=18,
                    timeout=1,
                )

    def test_reads_chat_meta_info(self):
        tokens, elapsed, accept, histogram = self.run_sample(
            response_payload(histogram=[1, 2, 3])
        )
        self.assertEqual(tokens, 18)
        self.assertEqual(elapsed, 2.0)
        self.assertEqual(accept, 2.25)
        self.assertEqual(histogram, [1, 2, 3])

    def test_falls_back_to_verify_count(self):
        _, _, accept, _ = self.run_sample(response_payload(accept=None, verify=6))
        self.assertEqual(accept, 3.0)

    def test_refuses_missing_speculative_metrics(self):
        with self.assertRaisesRegex(RuntimeError, "no spec_accept_length"):
            self.run_sample(response_payload(accept=None))

    def test_default_plan_is_exactly_n32(self):
        args = chat_bench.parse_args(["baseline"])
        self.assertEqual(args.reps * len(chat_bench.TASKS["open-ended"]), 32)
        self.assertEqual(chat_bench.selected_tasks("pooled-open"), ("code", "chat", "open-ended"))

    def test_derives_acceptance_by_position_from_histogram(self):
        samples = [
            chat_bench.Sample(
                task="open-ended",
                seed=index,
                repetition=0,
                completion_tokens=18,
                elapsed_seconds=1.0,
                tokens_per_second=18.0,
                accept_length=2.0,
                correct_drafts_histogram=[1, 2, 1],
            )
            for index in range(2)
        ]
        summary = chat_bench.summarize(samples)
        self.assertEqual(summary["correct_drafts_histogram"], [2, 4, 2])
        self.assertEqual(summary["verify_steps"], 8)
        self.assertEqual(summary["accept_by_position"][0]["accept_rate"], 0.75)
        self.assertEqual(summary["accept_by_position"][1]["accept_rate"], 0.25)

    def test_pads_unreached_block_tail_with_zero_rates(self):
        sample = chat_bench.Sample(
            task="open-ended",
            seed=0,
            repetition=0,
            completion_tokens=18,
            elapsed_seconds=1.0,
            tokens_per_second=18.0,
            accept_length=1.0,
            correct_drafts_histogram=[4],
        )
        summary = chat_bench.summarize([sample], expected_block_size=7)
        self.assertEqual(len(summary["accept_by_position"]), 7)
        self.assertEqual(summary["accept_by_position"][-1]["accept_rate"], 0.0)


class CompareABTests(unittest.TestCase):
    def arm(self, n=32, *, url="http://localhost:30000"):
        return {
            "plan": {
                "url": url,
                "model": "inkling-small",
                "tasks": ["open-ended"],
                "prompts_per_task": 4,
                "repetitions_per_prompt": 8,
                "samples_per_task": 32,
                "max_tokens": 160,
                "endpoint": "/v1/chat/completions",
            },
            "summaries": {"open-ended": {"n": n}},
        }

    def test_accepts_comparable_exact_n32(self):
        compare_ab.check_comparable(self.arm(), self.arm(), "open-ended")

    def test_rejects_wrong_sample_count(self):
        with self.assertRaisesRegex(ValueError, "exactly n=32"):
            compare_ab.check_comparable(self.arm(31), self.arm(), "open-ended")

    def test_rejects_endpoint_drift(self):
        with self.assertRaisesRegex(ValueError, "url"):
            compare_ab.check_comparable(
                self.arm(), self.arm(url="http://other.invalid"), "open-ended"
            )

    def guarded_arm(self, *, accept: float, accept_se: float, elapsed: float):
        arm = self.arm()
        arm["summaries"]["open-ended"].update(
            {"accept_length": {"mean": accept, "se": accept_se}}
        )
        arm["samples"] = [
            {"task": "open-ended", "elapsed_seconds": elapsed + index * 0.001}
            for index in range(32)
        ]
        return arm

    def test_e8_guard_rejects_accept_regression(self):
        baseline = self.guarded_arm(accept=2.2, accept_se=0.02, elapsed=6.0)
        candidate = self.guarded_arm(accept=2.0, accept_se=0.02, elapsed=6.0)
        failures, _ = compare_ab.regression_guards(
            baseline,
            candidate,
            "open-ended",
            require_accept=True,
            require_latency=False,
        )
        self.assertEqual(failures, ["accept loss is at least one combined standard error"])

    def test_e8_guard_rejects_latency_regression(self):
        baseline = self.guarded_arm(accept=2.2, accept_se=0.02, elapsed=6.0)
        candidate = self.guarded_arm(accept=2.2, accept_se=0.02, elapsed=6.5)
        failures, _ = compare_ab.regression_guards(
            baseline,
            candidate,
            "open-ended",
            require_accept=True,
            require_latency=True,
        )
        self.assertEqual(
            failures,
            ["latency increase is at least one combined standard error"],
        )

    def test_e8_guards_allow_nonregressing_candidate(self):
        baseline = self.guarded_arm(accept=2.2, accept_se=0.02, elapsed=6.0)
        candidate = self.guarded_arm(accept=2.21, accept_se=0.02, elapsed=5.9)
        failures, reports = compare_ab.regression_guards(
            baseline,
            candidate,
            "open-ended",
            require_accept=True,
            require_latency=True,
        )
        self.assertEqual(failures, [])
        self.assertEqual(len(reports), 2)


class RunnerDetachmentTests(unittest.TestCase):
    def test_every_nohup_launch_detaches_stdin(self):
        runners = sorted((ROOT / "scripts").glob("run-e*.sh"))
        self.assertTrue(runners)
        for runner in runners:
            commands = runner.read_text(encoding="utf-8").replace("\\\n", " ")
            for line_number, line in enumerate(
                commands.splitlines(), start=1
            ):
                if "nohup env" in line:
                    with self.subTest(runner=runner.name, line=line_number):
                        self.assertIn("</dev/null", line)

    def test_remote_nohup_launches_are_disowned(self):
        runners = sorted((ROOT / "scripts").glob("run-e*.sh"))
        for runner in runners:
            commands = runner.read_text(encoding="utf-8").replace("\\\n", " ")
            for line_number, line in enumerate(commands.splitlines(), start=1):
                if "ssh -o BatchMode=yes" in line and "nohup env" in line:
                    with self.subTest(runner=runner.name, line=line_number):
                        self.assertIn("disown", line)

    def test_remote_serving_launches_use_ssh_background_mode(self):
        runners = sorted((ROOT / "scripts").glob("run-e*.sh"))
        launches = []
        for runner in runners:
            commands = runner.read_text(encoding="utf-8").replace("\\\n", " ")
            for line_number, line in enumerate(commands.splitlines(), start=1):
                if "locked-experiment-launch.sh 1" in line:
                    launches.append((runner.name, line_number))
                    with self.subTest(runner=runner.name, line=line_number):
                        self.assertIn("ssh -f -o BatchMode=yes", line)
                        self.assertIn("exec env", line)
                        self.assertNotIn("nohup env", line)
        self.assertEqual(len(launches), 7)


class ConcurrencyBenchTests(unittest.TestCase):
    def test_default_plan_is_n32_at_every_level(self):
        args = concurrency_bench.parse_args(["baseline"])
        items = concurrency_bench.work_items(args.task, args.reps)
        self.assertEqual(len(items), 32)
        self.assertEqual(args.concurrency, [1, 2, 4, 8, 16])
        self.assertTrue(all(len(items) % level == 0 for level in args.concurrency))

    def test_refuses_partial_waves(self):
        with mock.patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                concurrency_bench.parse_args(["baseline", "--concurrency", "3"])

    def test_summarizes_aggregate_waves_separately(self):
        samples = [
            concurrency_bench.ConcurrentSample(
                concurrency=2,
                wave=index // 2,
                seed=index % 4,
                repetition=0,
                completion_tokens=10,
                elapsed_seconds=2.0,
                tokens_per_second=5.0,
                accept_length=2.0,
            )
            for index in range(4)
        ]
        waves = [
            concurrency_bench.Wave(2, 0, 2, 20, 2.0, 10.0),
            concurrency_bench.Wave(2, 1, 2, 20, 1.0, 20.0),
        ]
        summary = concurrency_bench.summarize(samples, waves)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["wave_n"], 2)
        self.assertEqual(summary["aggregate_tokens_per_second"]["mean"], 15.0)


class SelectBlockSizeTests(unittest.TestCase):
    def arm(self, block: int, mean: float, se: float):
        return {
            "plan": {
                "url": "http://localhost:30000",
                "model": "inkling-small",
                "tasks": ["open-ended"],
                "prompts_per_task": 4,
                "repetitions_per_prompt": 8,
                "samples_per_task": 32,
                "max_tokens": 160,
                "endpoint": "/v1/chat/completions",
                "block_size": block,
            },
            "summaries": {
                "open-ended": {
                    "n": 32,
                    "histogram_complete": True,
                    "accept_by_position": [
                        {"position": position, "accept_rate": 0.5 / position}
                        for position in range(1, block + 1)
                    ],
                    "tokens_per_second": {"mean": mean, "se": se},
                }
            },
        }

    def test_selects_nonoverlapping_half_token_gain(self):
        by_block = select_block_size.validate(
            [self.arm(5, 24.0, 0.1), self.arm(6, 23.0, 0.1), self.arm(7, 23.0, 0.1)],
            "open-ended",
        )
        decision = select_block_size.decide(by_block, "open-ended", 0.5)
        self.assertEqual(decision["verdict"], "ACCEPT")
        self.assertEqual(decision["accepted_block_size"], 5)

    def test_retains_seven_when_gain_is_not_proven(self):
        by_block = select_block_size.validate(
            [self.arm(5, 23.4, 0.3), self.arm(6, 23.3, 0.3), self.arm(7, 23.0, 0.3)],
            "open-ended",
        )
        decision = select_block_size.decide(by_block, "open-ended", 0.5)
        self.assertEqual(decision["verdict"], "RETAIN_BLOCK_7")
        self.assertIsNone(decision["accepted_block_size"])

    def test_rejects_incomplete_position_evidence(self):
        short = self.arm(7, 23.0, 0.1)
        short["summaries"]["open-ended"]["accept_by_position"] = short["summaries"][
            "open-ended"
        ]["accept_by_position"][:-1]
        with self.assertRaisesRegex(ValueError, "only 6 draft positions"):
            select_block_size.validate(
                [self.arm(5, 24.0, 0.1), self.arm(6, 23.0, 0.1), short],
                "open-ended",
            )


class EvaluateBootCacheTests(unittest.TestCase):
    def test_summarizes_exactly_three_boots(self):
        summary = evaluate_boot_cache.summarize_timings(
            [{"time_to_t4_seconds": value} for value in (200.0, 210.0, 220.0)]
        )
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean"], 210.0)
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            evaluate_boot_cache.summarize_timings(
                [{"time_to_t4_seconds": 200.0}, {"time_to_t4_seconds": 210.0}]
            )

    def test_accepts_material_sub_four_minute_warm_boot(self):
        verdict, _ = evaluate_boot_cache.decision(
            480.0,
            220.0,
            {"mean": 23.9, "se": 0.3},
            {"mean": 23.8, "se": 0.3},
        )
        self.assertEqual(verdict, "ACCEPT")

    def test_rejects_throughput_regression_despite_fast_boot(self):
        verdict, _ = evaluate_boot_cache.decision(
            480.0,
            200.0,
            {"mean": 23.9, "se": 0.2},
            {"mean": 23.5, "se": 0.2},
        )
        self.assertEqual(verdict, "REJECT")

    def test_does_not_accept_small_boot_change(self):
        verdict, _ = evaluate_boot_cache.decision(
            260.0,
            230.0,
            {"mean": 23.9, "se": 0.3},
            {"mean": 23.9, "se": 0.3},
        )
        self.assertEqual(verdict, "INCONCLUSIVE")


class EvaluateMemfracTests(unittest.TestCase):
    def result(self, c8: tuple[float, float], c16: tuple[float, float]):
        return {
            "plan": {
                "url": "http://localhost:30000",
                "model": "inkling-small",
                "task": "open-ended",
                "prompts": 4,
                "repetitions_per_prompt": 8,
                "samples_per_concurrency": 32,
                "concurrencies": [8, 16],
                "max_tokens": 160,
                "warmups": 2,
                "endpoint": "/v1/chat/completions",
            },
            "summaries": {
                "8": {"aggregate_tokens_per_second": {"mean": c8[0], "se": c8[1]}},
                "16": {"aggregate_tokens_per_second": {"mean": c16[0], "se": c16[1]}},
            },
        }

    def test_accepts_lower_fraction_only_for_evidenced_baseline_memory_failure(self):
        result = evaluate_memfrac.decide(
            {"stable": False, "memory_event": True},
            {"stable": True, "memory_event": False},
            None,
            self.result((80.0, 2.0), (100.0, 3.0)),
        )
        self.assertEqual(result["verdict"], "ACCEPT_0.68")

    def test_retains_safe_baseline(self):
        baseline = self.result((80.0, 2.0), (100.0, 3.0))
        candidate = self.result((80.5, 2.0), (100.5, 3.0))
        result = evaluate_memfrac.decide(
            {"stable": True, "memory_event": False},
            {"stable": True, "memory_event": False},
            baseline,
            candidate,
        )
        self.assertEqual(result["verdict"], "RETAIN_0.85")

    def test_invalidates_unevidenced_baseline_failure(self):
        result = evaluate_memfrac.decide(
            {"stable": False, "memory_event": False},
            {"stable": True, "memory_event": False},
            None,
            self.result((80.0, 2.0), (100.0, 3.0)),
        )
        self.assertEqual(result["verdict"], "INVALID")


class GSM8KEvalTests(unittest.TestCase):
    def test_prefers_hash_answer_and_normalizes_commas(self):
        predicted, method = gsm8k_eval.extract_prediction(
            "I considered 19, then corrected it.\n#### $1,234"
        )
        self.assertEqual(predicted, gsm8k_eval.Decimal("1234"))
        self.assertEqual(method, "hash")

    def test_uses_last_numeric_fallback_explicitly(self):
        predicted, method = gsm8k_eval.extract_prediction("First 4, final answer is 7")
        self.assertEqual(predicted, gsm8k_eval.Decimal("7"))
        self.assertEqual(method, "last-number-fallback")


class NIAHEvalTests(unittest.TestCase):
    def test_calibrates_record_count_against_measured_tokens(self):
        records, tokens = niah_eval.calibrate_record_count(
            target_tokens=1000,
            tolerance_tokens=1,
            count_for_records=lambda count: 100 + 10 * count,
        )
        self.assertEqual(records, 90)
        self.assertEqual(tokens, 1000)

    def test_places_deterministic_unique_needle(self):
        secret = niah_eval.secret_for(512_000, 0.5, 0)
        messages, needle_index = niah_eval.build_messages(
            record_count=10,
            depth=0.5,
            secret=secret,
            seed=0,
        )
        self.assertEqual(needle_index, 5)
        self.assertEqual(messages[0]["content"].count(secret), 1)
        self.assertEqual(secret, niah_eval.secret_for(512_000, 0.5, 0))


class ToolCallRegressionTests(unittest.TestCase):
    def test_detects_every_forbidden_parser_token(self):
        content = " ".join(tool_call_regression.FORBIDDEN_TOKENS)
        self.assertEqual(
            tool_call_regression.leaked_tokens(content),
            list(tool_call_regression.FORBIDDEN_TOKENS),
        )

    def test_accepts_structured_expected_tool_call(self):
        case = tool_call_regression.CASES[0]
        message = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": case["name"],
                        "arguments": json.dumps(case["required_arguments"]),
                    },
                }
            ],
        }
        call, failures = tool_call_regression.validate_tool_message(message, case)
        self.assertEqual(call["id"], "call-1")
        self.assertEqual(failures, [])


class RepoFingerprintTests(unittest.TestCase):
    def test_includes_untracked_payload_and_excludes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "scripts" / "runner.sh"
            source.parent.mkdir()
            source.write_text("first\n", encoding="utf-8")
            source.chmod(0o755)
            initial = repo_fingerprint.fingerprint(root)

            artifact = root / "artifacts" / "result.json"
            artifact.parent.mkdir()
            artifact.write_text("{}\n", encoding="utf-8")
            self.assertEqual(initial["sha256"], repo_fingerprint.fingerprint(root)["sha256"])

            source.write_text("second\n", encoding="utf-8")
            self.assertNotEqual(initial["sha256"], repo_fingerprint.fingerprint(root)["sha256"])

    def test_executable_mode_affects_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "runner.sh"
            path.write_text("#!/bin/bash\n", encoding="utf-8")
            path.chmod(0o644)
            first = repo_fingerprint.fingerprint(root)["sha256"]
            path.chmod(0o755)
            self.assertNotEqual(first, repo_fingerprint.fingerprint(root)["sha256"])


if __name__ == "__main__":
    unittest.main()
