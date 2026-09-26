import unittest

from flotilla import util
from flotilla.config import interpolate_env


class ThinkTests(unittest.TestCase):
    def test_strip_block(self):
        content, reasoning = util.strip_think("<think>step 1\nstep 2</think>\n\nThe answer is 4.")
        self.assertEqual(content, "The answer is 4.")
        self.assertEqual(reasoning, "step 1\nstep 2")

    def test_unclosed_block_is_all_reasoning(self):
        content, reasoning = util.strip_think("<think>still thinking when tokens ran out")
        self.assertEqual(content, "")
        self.assertIn("still thinking", reasoning)

    def test_stray_closing_tag(self):
        content, reasoning = util.strip_think("reasoning opened by the template</think>Final.")
        self.assertEqual(content, "Final.")
        self.assertEqual(reasoning, "reasoning opened by the template")

    def test_no_tags(self):
        self.assertEqual(util.strip_think("plain"), ("plain", ""))
        self.assertEqual(util.strip_think(None), ("", ""))

    def test_stream_filter_split_tags(self):
        text = "<think>hidden part</think>Visible answer<thinking>more</thinking> end"
        for size in (1, 2, 3, 5, 7, 100):
            f = util.ThinkStreamFilter()
            content, reasoning = [], []
            for i in range(0, len(text), size):
                c, r = f.feed(text[i:i + size])
                content.append(c)
                reasoning.append(r)
            c, r = f.flush()
            content.append(c)
            reasoning.append(r)
            self.assertEqual("".join(content), "Visible answer end", size)
            self.assertEqual("".join(reasoning), "hidden partmore", size)

    def test_stream_filter_keeps_lookalike_text(self):
        f = util.ThinkStreamFilter()
        out = f.feed("a < b and <thin")[0] + f.feed("gs>")[0] + f.flush()[0]
        self.assertEqual(out, "a < b and <things>")


class ParsingTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(util.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(util.extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nDone.'), {"a": [1, 2]})
        self.assertEqual(util.extract_json('The plan: {"route": "deep"} as requested'), {"route": "deep"})
        self.assertEqual(util.extract_json('<think>{"no": 1}</think>{"yes": 2}'), {"yes": 2})
        self.assertIsNone(util.extract_json("no json here"))
        self.assertIsNone(util.extract_json(""))

    def test_parse_ranking(self):
        labels = ["A", "B", "C"]
        self.assertEqual(util.parse_ranking("blah\nFINAL RANKING: C, A, B", labels), ["C", "A", "B"])
        self.assertEqual(
            util.parse_ranking("FINAL RANKING:\n1. Response B\n2. Response C\n3. Response A\nA is weak.", labels),
            ["B", "C", "A"],
        )
        # missing labels are appended, unknown ones ignored
        self.assertEqual(util.parse_ranking("FINAL RANKING: B, Z", labels), ["B", "A", "C"])
        self.assertEqual(util.parse_ranking("no ranking at all", labels), ["A", "B", "C"])

    def test_parse_choice(self):
        labels = ["A", "B", "C"]
        self.assertEqual(util.parse_choice("B is better.\nBEST: B", labels), "B")
        self.assertEqual(util.parse_choice("BEST: **Candidate C**", labels), "C")
        self.assertEqual(util.parse_choice("best: a", labels), "A")
        self.assertIsNone(util.parse_choice("I like the second one", labels))
        self.assertIsNone(util.parse_choice("BEST: Q", labels))

    def test_is_approval(self):
        self.assertTrue(util.is_approval("APPROVED"))
        self.assertTrue(util.is_approval("**APPROVED**"))
        self.assertTrue(util.is_approval("Approved."))
        self.assertTrue(util.is_approval("<think>fine</think>APPROVED"))
        self.assertFalse(util.is_approval("- missing an example\n- APPROVED otherwise"))
        self.assertFalse(util.is_approval("Not approved: the maths is wrong"))
        self.assertFalse(util.is_approval(""))

    def test_normalize_answer(self):
        self.assertEqual(util.normalize_answer("Answer: 42."), util.normalize_answer("42"))
        self.assertEqual(util.normalize_answer("Paris!"), util.normalize_answer("  paris "))


class MessageTests(unittest.TestCase):
    def test_render_single_and_multi_turn(self):
        self.assertEqual(util.render_conversation([{"role": "user", "content": "hi"}]), "hi")
        text = util.render_conversation([
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "What is RAID?"},
            {"role": "assistant", "content": "Disk redundancy."},
            {"role": "user", "content": "Which level for 4 disks?"},
        ])
        self.assertIn("User: What is RAID?", text)
        self.assertIn("Assistant: Disk redundancy.", text)
        self.assertTrue(text.endswith("Latest request:\nWhich level for 4 disks?"))
        self.assertNotIn("be nice", text)

    def test_content_parts(self):
        msg = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        self.assertEqual(util.content_to_text(msg), "look\n[image]")

    def test_system_merging(self):
        msgs = [{"role": "system", "content": "client rules"}, {"role": "user", "content": "q"}]
        merged = util.with_system(msgs, "role instructions")
        self.assertEqual(merged[0], {"role": "system", "content": "client rules\n\nrole instructions"})
        self.assertEqual(len(merged), 2)
        pre = util.prepend_system(merged, "persona")
        self.assertTrue(pre[0]["content"].startswith("persona\n\nclient rules"))
        self.assertEqual(util.with_system([{"role": "user", "content": "q"}]), [{"role": "user", "content": "q"}])


class MiscTests(unittest.TestCase):
    def test_canonical_model(self):
        self.assertEqual(util.canonical_model("Llama3.2:latest"), "llama3.2")
        self.assertEqual(util.canonical_model("qwen3.5:4b"), "qwen3.5:4b")

    def test_labels_and_lists(self):
        self.assertEqual(util.parse_labels("gpu=nvidia, fast"), {"gpu": "nvidia", "fast": "true"})
        self.assertEqual(util.parse_list("a, b;c,,"), ["a", "b", "c"])

    def test_bearer(self):
        self.assertEqual(util.bearer_token({"authorization": "Bearer abc"}), "abc")
        self.assertEqual(util.bearer_token({"x-api-key": "k"}), "k")
        self.assertIsNone(util.bearer_token({}))
        self.assertTrue(util.consteq("a", "a"))
        self.assertFalse(util.consteq("a", None))

    def test_interpolate_env(self):
        env = {"A": "1", "EMPTY": ""}
        self.assertEqual(interpolate_env("x=${A} y=${B:-two} z=${EMPTY:-d} w=${MISSING}", env), "x=1 y=two z=d w=")


if __name__ == "__main__":
    unittest.main()
