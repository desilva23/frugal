"""Tests for query reformulation.

Generating variants is easy; the value is in refusing to issue ones that would
return the same page. Most of this file tests the refusal, and the engine-shaping
tests exist because sending the wrong shape wastes the search however well it is
worded.
"""

from __future__ import annotations

import pytest

from frugal.reformulate import (
    expand_query,
    feedback_terms,
    novelty,
    reformulate,
    shape_for_engine,
    similarity,
    tokenise,
)


def queries(*args: object, **kwargs: object) -> list[str]:
    return [r.query for r in reformulate(*args, **kwargs)]  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


def test_stopwords_are_dropped() -> None:
    assert tokenise("What is the price of a laptop?") == ("price", "laptop")


def test_tokens_keep_their_order_without_duplicates() -> None:
    assert tokenise("solar panel solar cell") == ("solar", "panel", "cell")


def test_hyphenated_words_survive() -> None:
    assert "peer-reviewed" in tokenise("peer-reviewed studies")


def test_single_characters_are_dropped() -> None:
    assert tokenise("a b cd") == ("cd",)


def test_case_is_normalised() -> None:
    assert tokenise("Solar PANEL") == ("solar", "panel")


# --------------------------------------------------------------------------
# Similarity and novelty
# --------------------------------------------------------------------------


def test_identical_queries_are_maximally_similar() -> None:
    assert similarity("solar panel efficiency", "solar panel efficiency") == 1.0


def test_stopword_only_differences_are_identical() -> None:
    """Engines discard stopwords, so these are the same query."""
    assert similarity("price of a laptop", "the price of laptop") == 1.0


def test_unrelated_queries_share_nothing() -> None:
    assert similarity("solar panels", "coffee shops") == 0.0


def test_novelty_is_total_when_nothing_was_issued() -> None:
    assert novelty("anything", ()) == 1.0


def test_novelty_is_zero_against_an_identical_prior() -> None:
    assert novelty("solar panels", ("solar panels",)) == 0.0


def test_novelty_measures_against_the_closest_prior() -> None:
    scored = novelty("solar panel cost", ("coffee shops", "solar panel cost"))
    assert scored == 0.0


# --------------------------------------------------------------------------
# Refusing redundant queries
# --------------------------------------------------------------------------


def test_a_query_already_issued_is_not_returned_again() -> None:
    question = "What is the latest news on railway electrification?"
    first = queries(question, engine="google", limit=3)
    again = queries(question, engine="google", issued=tuple(first), limit=3)
    assert not set(again) & set(first)


def test_a_plan_eventually_runs_out_of_distinct_angles() -> None:
    """The signal that a plan should stop rather than keep paying."""
    question = "What is the latest news on railway electrification?"
    issued: tuple[str, ...] = ()
    for _ in range(5):
        batch = queries(question, engine="google", issued=issued, limit=2)
        if not batch:
            break
        issued += tuple(batch)
    else:
        pytest.fail("reformulation never ran out of ideas")
    assert issued


def test_variants_within_one_call_are_distinct_from_each_other() -> None:
    results = reformulate("cheapest wireless earbuds under 3000", engine="google", limit=4)
    assert len({r.query for r in results}) == len(results)


def test_a_high_novelty_floor_suppresses_everything_but_the_first() -> None:
    results = reformulate("solar panel efficiency records", engine="google", min_novelty=0.99)
    assert len(results) == 1


def test_a_zero_floor_keeps_every_candidate() -> None:
    lenient = reformulate("latest solar panel efficiency records", engine="google",
                          limit=10, min_novelty=0.0)
    strict = reformulate("latest solar panel efficiency records", engine="google", limit=10)
    assert len(lenient) >= len(strict)


# --------------------------------------------------------------------------
# Engine shaping
# --------------------------------------------------------------------------


def test_trends_gets_a_bare_term_not_a_sentence() -> None:
    """Trends matches a term; a full question returns nothing useful."""
    question = "Is interest in electric vehicles growing in India over time?"
    first = reformulate(question, engine="google_trends")[0]
    assert first.strategy == "bare-term"
    assert len(first.query.split()) <= 3
    assert "electric" in first.query and "vehicles" in first.query


def test_framing_words_are_not_mistaken_for_the_subject() -> None:
    """"is interest in X growing" is a question about X, not about interest."""
    question = "Is interest in electric vehicles growing over time?"
    assert "interest" not in reformulate(question, engine="google_trends")[0].query


def test_shopping_strips_question_framing() -> None:
    first = reformulate("What is the cheapest laptop for students?", engine="google_shopping")[0]
    assert first.strategy == "product-terms"
    assert "what" not in first.query.lower()


def test_each_shaped_engine_declares_a_reason() -> None:
    for engine in ["google_trends", "google_scholar", "google_shopping", "google_jobs",
                   "google_maps", "google_patents"]:
        shaped = shape_for_engine("some question about things", engine)
        assert shaped is not None
        _, _, rationale = shaped
        assert rationale


def test_web_search_needs_no_special_shape() -> None:
    assert shape_for_engine("anything", "google") is None


def test_the_shaped_query_leads_for_an_engine_that_needs_it() -> None:
    """Shaping is the highest-value variant, so it must not be crowded out."""
    results = reformulate("Who patented graphene transistors?", engine="google_patents", limit=1)
    assert results[0].strategy == "invention-terms"


# --------------------------------------------------------------------------
# Output contract
# --------------------------------------------------------------------------


def test_limit_is_respected() -> None:
    assert len(reformulate("latest cheapest solar panel research", engine="google", limit=2)) == 2


def test_every_result_carries_its_reasoning() -> None:
    for result in reformulate("cheapest solar panels in India", engine="google_shopping"):
        assert result.strategy and result.rationale
        assert 0.0 <= result.novelty <= 1.0
        assert result.query.strip() == result.query


def test_explain_is_readable() -> None:
    explanation = reformulate("solar panel cost", engine="google")[0].explain()
    assert "novelty" in explanation


def test_reformulation_is_deterministic() -> None:
    question = "What is the cheapest electric scooter and is interest growing?"
    first = queries(question, engine="google", limit=3)
    for _ in range(10):
        assert queries(question, engine="google", limit=3) == first


def test_an_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty question"):
        reformulate("   ")


def test_a_question_that_is_entirely_framing_still_yields_a_query() -> None:
    """Falling through to nothing would leave the plan with no query at all."""
    assert reformulate("what is the best?", engine="google_trends")


# --------------------------------------------------------------------------
# Pseudo-relevance feedback
# --------------------------------------------------------------------------
#
# A later round that rewords the question is not adaptive; it just costs a
# second search for the same page. A later round that asks about vocabulary the
# first round *found* is asking something the question could not have.


def test_terms_frequent_across_results_are_returned() -> None:
    texts = ["quantum computing breakthrough"] * 3 + ["unrelated page"]
    assert "quantum" in feedback_terms(texts, "what is the latest breakthrough?")


def test_terms_the_question_already_asked_are_not_returned() -> None:
    texts = ["solar panel efficiency"] * 4
    assert "solar" not in feedback_terms(texts, "solar panel efficiency records")


def test_a_plural_of_a_question_term_is_not_new_vocabulary() -> None:
    """"launches" for a question about a "launch" costs a search and buys nothing."""
    texts = ["isro launches satellite", "isro launches rocket", "isro launches probe"]
    assert "launches" not in feedback_terms(texts, "What is the latest ISRO mission launch?")


def test_a_possessive_of_a_question_term_is_not_new_vocabulary() -> None:
    texts = ["india's semiconductor plan"] * 4
    assert "india's" not in feedback_terms(texts, "semiconductor manufacturing in India")


def test_hyphenated_question_terms_are_matched_by_their_parts() -> None:
    """"CRISPR-Cas9" is one token here and two in the index."""
    texts = ["crispr cas9 editing"] * 4
    terms = feedback_terms(texts, "Which paper introduced CRISPR-Cas9 genome editing?")
    assert "crispr" not in terms
    assert "cas9" not in terms


def test_a_term_in_one_result_only_is_not_characteristic() -> None:
    """One verbose page must not nominate its own vocabulary."""
    texts = ["common topic here", "common topic here", "idiosyncratic tangent"]
    assert "idiosyncratic" not in feedback_terms(texts, "what about the topic?")


def test_boilerplate_is_excluded() -> None:
    texts = ["read more on our website click here"] * 5
    assert not set(feedback_terms(texts, "a question")) & {"read", "website", "click", "here"}


def test_no_results_yields_no_terms() -> None:
    assert feedback_terms([], "anything") == ()


def test_feedback_is_deterministic() -> None:
    """A plan that adapts must still reproduce exactly."""
    texts = ["alpha beta", "alpha gamma", "alpha beta", "delta alpha"]
    first = feedback_terms(texts, "a question")
    for _ in range(10):
        assert feedback_terms(texts, "a question") == first


def test_limit_caps_the_terms_returned() -> None:
    texts = ["alpha beta gamma delta epsilon"] * 4
    assert len(feedback_terms(texts, "a question", limit=2)) == 2


def test_expansion_appends_only_genuinely_new_terms() -> None:
    assert expand_query("solar panels", ("efficiency",)) == "solar panels efficiency"
    assert expand_query("solar panels", ("panel",)) == "solar panels"


def test_expansion_with_nothing_to_add_is_unchanged() -> None:
    assert expand_query("solar panels", ()) == "solar panels"
