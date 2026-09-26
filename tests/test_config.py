import os
import textwrap
import unittest
from pathlib import Path

from flotilla.config import ConfigError, load_config, resolve_member, team_models

ROOT = Path(__file__).resolve().parents[1]

BASE = """
members:
  a: {model: small-a}
  b: {model: [big-b, small-b], temperature: 0.1, system: "You are B."}
"""


def cfg(text: str):
    return load_config(text=textwrap.dedent(BASE) + textwrap.dedent(text))


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = load_config(ROOT / "config" / "flotilla.yaml")
        self.assertIn("moa", config.teams)
        self.assertEqual(config.teams["auto"].strategy, "route")
        self.assertIn("qwen3.5:4b", team_models(config, "auto"))

    def test_example_configs_are_valid(self):
        for path in sorted((ROOT / "config" / "examples").glob("*.yaml")):
            with self.subTest(path=path.name):
                load_config(path)

    def test_defaults_merge(self):
        c = cfg("""
        defaults: {temperature: 0.7, max_tokens: 100, labels: {zone: home}}
        teams:
          t: {strategy: single, member: b}
        """)
        res = resolve_member(c, "b")
        self.assertEqual(res.candidates, ["big-b", "small-b"])
        self.assertEqual(res.params.temperature, 0.1)      # member wins
        self.assertEqual(res.params.max_tokens, 100)       # default applies
        self.assertEqual(res.params.labels, {"zone": "home"})

    def test_inline_member_and_team_ref(self):
        c = cfg("""
        teams:
          inner: {strategy: single, member: a}
          outer:
            strategy: mixture
            proposers: [a, {model: inline-model, temperature: 0.3}, "team:inner"]
            aggregator: b
        """)
        self.assertEqual(team_models(c, "outer"), {"small-a", "inline-model", "big-b", "small-b"})
        self.assertEqual(resolve_member(c, "team:inner").team, "inner")

    def test_unknown_member(self):
        with self.assertRaises(ConfigError) as ctx:
            cfg("""
            teams:
              t: {strategy: mixture, proposers: [a, ghost], aggregator: b}
            """)
        self.assertIn("unknown member 'ghost'", str(ctx.exception))

    def test_unknown_team(self):
        with self.assertRaises(ConfigError) as ctx:
            cfg("""
            teams:
              t: {strategy: single, member: "team:nope"}
            """)
        self.assertIn("unknown team 'nope'", str(ctx.exception))

    def test_cycle(self):
        with self.assertRaises(ConfigError) as ctx:
            cfg("""
            teams:
              x: {strategy: single, member: "team:y"}
              y:
                strategy: route
                router: a
                routes: [{name: back, target: "team:x"}]
            """)
        self.assertIn("team cycle", str(ctx.exception))

    def test_strategy_requirements(self):
        cases = {
            "mixture without aggregator": "t: {strategy: mixture, proposers: [a]}",
            "vote with one candidate": "t: {strategy: vote, voters: [a]}",
            "route default missing": "t: {strategy: route, router: a, default: zzz, routes: [{name: r, target: a}]}",
            "unknown strategy": "t: {strategy: telepathy, member: a}",
            "unknown field": "t: {strategy: single, member: a, colour: blue}",
        }
        for name, team in cases.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                cfg("teams:\n  " + team)

    def test_member_needs_model_or_team(self):
        with self.assertRaises(ConfigError):
            load_config(text="members:\n  x: {temperature: 1}\n")

    def test_env_interpolation(self):
        os.environ["FLOTILLA_TEST_MODEL"] = "env-model"
        try:
            c = load_config(text="members:\n  m: {model: '${FLOTILLA_TEST_MODEL}'}\n"
                                 "teams:\n  t: {strategy: single, member: m}\n")
            self.assertEqual(resolve_member(c, "m").candidates, ["env-model"])
        finally:
            del os.environ["FLOTILLA_TEST_MODEL"]

    def test_bad_yaml(self):
        with self.assertRaises(ConfigError):
            load_config(text="teams: [unclosed")


if __name__ == "__main__":
    unittest.main()
