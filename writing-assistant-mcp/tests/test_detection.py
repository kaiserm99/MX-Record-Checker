from writing_assistant_mcp.classifier import windowed_scores
from writing_assistant_mcp.compare import compare, meaning_check
from writing_assistant_mcp.detector import detect
from writing_assistant_mcp.plan import build_plan
from writing_assistant_mcp.tells import SENTENCE_MATCHERS


def tells_in(sentence):
    return {tid for tid, match in SENTENCE_MATCHERS.items() if match(sentence)}


def test_sentence_tells():
    assert tells_in("Moreover, this pivotal study underscores the point.") == {
        "transition_opener",
        "ai_vocabulary",
    }
    assert "stock_phrase" in tells_in("In today's fast-paced world, speed matters.")
    assert "negative_parallelism" in tells_in("It is not just a tool, but a mindset.")
    assert "participle_tail" in tells_in("Sales rose 5%, highlighting strong demand.")
    assert "em_dash" in tells_in("The result—surprisingly—held.")
    assert "vague_attribution" in tells_in("Experts suggest that this works.")
    assert "summary_opener" in tells_in("In conclusion, it works.")
    assert tells_in("I moved my desk upstairs and it worked.") == set()


def test_participle_in_a_list_is_not_a_tail():
    assert "participle_tail" not in tells_in(
        "By embracing flexibility, fostering trust, and investing in tools, firms win."
    )


def test_heuristics_separate_ai_and_human_samples(sample):
    ai = detect(sample("ai"), classifier=None)
    human = detect(sample("human"), classifier=None)
    assert ai["overall"]["band"] == "likely_ai"
    assert human["overall"]["band"] == "likely_human"
    assert ai["overall"]["confidence"] == "low"  # heuristics only
    assert any(d["id"] == "uniform_sentence_length" for d in ai["document_tells"])
    assert ai["top_reasons"][0]["tell"] == "ai_vocabulary"


def test_windowed_scores_cover_every_sentence(fake_classifier):
    sentences = [f"Sentence {i}." for i in range(7)] + ["Moreover, done."]
    document, per_sentence = windowed_scores(fake_classifier, sentences, window=4, stride=2)
    assert len(per_sentence) == 8
    assert per_sentence[0] == 0.1 and per_sentence[-1] == 0.9
    windows = fake_classifier.calls[0][:4]
    assert windows[-1].endswith("Moreover, done.")  # tail window added
    assert 0.1 < document <= 0.9


def test_classifier_is_blended_and_explains_untagged_flags(fake_classifier):
    text = "Moreover, the plan works. It ships today. We tested it well."
    result = detect(text, classifier=fake_classifier)
    assert result["engines"]["classifier"]["model"] == "fake"
    flagged = [s for s in result["sentences"] if s["flagged"]]
    assert flagged
    untagged = [s for s in flagged if s["index"] == 1][0]
    assert [t["id"] for t in untagged["tells"]] == ["classifier_pattern"]


def test_rewrite_plan(sample):
    plan = build_plan(detect(sample("ai"), classifier=None))
    assert plan["status"] == "revise"
    high = [s for s in plan["sentences"] if s["priority"] == "high"]
    assert high and high == sorted(high, key=lambda s: -s["ai_score"])
    used = {p["tell"] for s in plan["sentences"] for p in s["problems"]}
    assert used <= set(plan["tell_guide"])
    assert all({"why", "fix"} <= set(g) for g in plan["tell_guide"].values())

    done = build_plan(detect(sample("human"), classifier=None))
    assert done["status"] == "done"


def test_compare_versions(sample):
    before_text, after_text = sample("ai"), sample("ai_rev1")
    result = compare(
        before_text, after_text, detect(before_text, classifier=None), detect(after_text, classifier=None)
    )
    assert result["improved"] and result["delta"] < -0.3
    assert result["tells_resolved"]["ai_vocabulary"] > 5
    assert result["meaning_check"]["warnings"] == []
    assert result["recommendation"].startswith("Done")


def test_meaning_check_flags_dropped_facts():
    check = meaning_check("Revenue grew 12% in Berlin during 2024.", "Revenue grew a lot.")
    assert check["missing_numbers"] == ["12%", "2024"]
    assert "Berlin" in check["missing_names"]
    assert check["warnings"]


def test_short_text_gets_no_verdict():
    result = detect("Hello there. This is a test.", classifier=None)
    assert result["overall"]["band"] == "too_short"
    assert result["overall"]["confidence"] == "low"
    assert any("unreliable" in n for n in result["notes"])
