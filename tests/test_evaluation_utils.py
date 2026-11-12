# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from evaluation.benchmarks.aime25.calculate_metrics import score_aime as score_aime25
from evaluation.benchmarks.math500.calculate_metrics import score_aime as score_math500
from evaluation.benchmarks.utils import extract_boxed


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"Answer: \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"Answer: \boxed{\left\{x \mid x > 0\right\}}", r"\left\{x \mid x > 0\right\}"),
    ],
)
def test_extract_boxed_handles_nested_latex(text, expected):
    assert extract_boxed(text) == expected


def test_extract_boxed_selects_first_or_last_answer():
    text = r"Draft: \boxed{1}. Final: \boxed{2}."

    assert extract_boxed(text) == "1"
    assert extract_boxed(text, last=True) == "2"


@pytest.mark.parametrize(
    "text",
    [
        "No boxed answer",
        r"Incomplete: \boxed{\frac{1}{2}",
    ],
)
def test_extract_boxed_returns_none_without_complete_box(text):
    assert extract_boxed(text) is None


def test_aime25_scores_last_boxed_answer():
    assert score_aime25(r"Draft: \boxed{1}. Final: \boxed{\frac{1}{2}}.", r"\frac{1}{2}")


def test_math500_scores_first_boxed_answer():
    assert score_math500(r"First: \boxed{\frac{1}{2}}. Later: \boxed{1}.", r"\frac{1}{2}")


@pytest.mark.parametrize("predicted_answer", ["", " \n", "..."])
def test_needle_in_haystack_answer_without_words_scores_zero(predicted_answer):
    pytest.importorskip("rouge")
    from evaluation.benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics

    df = pd.DataFrame({"needle": ["The secret number is 42"], "predicted_answer": [predicted_answer]})
    zero = {"r": 0.0, "p": 0.0, "f": 0.0}

    assert calculate_metrics(df) == [{"rouge-1": zero, "rouge-2": zero, "rouge-l": zero}]
