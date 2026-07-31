"""Tests for the public domain contract."""

import re

import pytest

from topobench import SUPPORTED_DOMAINS, require_supported_domain
from topobench.domains import __all__ as domain_exports


def test_supported_domains_are_closed_and_ordered():
    """The public domain set has an explicit, stable order."""
    assert SUPPORTED_DOMAINS == ("graph", "heterogeneous", "hypergraph")


@pytest.mark.parametrize("domain", SUPPORTED_DOMAINS)
def test_supported_domains_are_returned_unchanged(domain):
    """Every supported domain passes through validation unchanged."""
    assert require_supported_domain(domain) == domain


@pytest.mark.parametrize(
    "domain",
    ["cell", "simplicial", "combinatorial", "pointcloud"],
)
def test_legacy_domains_are_rejected(domain):
    """Legacy domains are outside the closed public contract."""
    expected = f"Unsupported domain {domain!r}; expected one of {SUPPORTED_DOMAINS}"
    with pytest.raises(ValueError, match=re.escape(expected)):
        require_supported_domain(domain)


@pytest.mark.parametrize("domain", [None, 1, ("graph",)])
def test_non_string_domains_are_rejected(domain):
    """Domain validation rejects non-string inputs explicitly."""
    with pytest.raises(TypeError, match="domain must be a string"):
        require_supported_domain(domain)


def test_domain_module_exports_only_the_public_contract():
    """The domain module exposes no additional public symbols."""
    assert domain_exports == ["SUPPORTED_DOMAINS", "require_supported_domain"]
