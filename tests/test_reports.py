"""Tests for beer.reports.css — make_style_tag()."""
from __future__ import annotations
import pytest
from beer.reports.css import make_style_tag, REPORT_CSS


def test_make_style_tag_returns_string():
    tag = make_style_tag()
    assert isinstance(tag, str)


def test_make_style_tag_is_html_style_element():
    tag = make_style_tag()
    assert tag.startswith("<style>"), f"Expected '<style>' prefix, got: {tag[:20]!r}"
    assert tag.endswith("</style>"), f"Expected '</style>' suffix, got: {tag[-20:]!r}"


def test_make_style_tag_contains_css(fus_lc):
    tag = make_style_tag()
    # Should contain some CSS selectors
    assert "body" in tag
    assert "font-family" in tag


def test_make_style_tag_default_accent():
    """Default accent colour must appear in REPORT_CSS."""
    assert "#4361ee" in REPORT_CSS


def test_make_style_tag_custom_accent():
    custom_accent = "#ff0000"
    tag = make_style_tag(accent=custom_accent)
    assert custom_accent in tag


def test_make_style_tag_accent_replaces_default():
    default_tag = make_style_tag()
    custom_tag  = make_style_tag(accent="#abcdef")
    # The default accent should not appear in the custom tag
    assert "#4361ee" not in custom_tag
    # But default tag should have it
    assert "#4361ee" in default_tag


def test_report_css_is_non_empty_string():
    assert isinstance(REPORT_CSS, str)
    assert len(REPORT_CSS) > 0
