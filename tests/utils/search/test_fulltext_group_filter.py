"""Unit tests for the group filter in both fulltext query builders.

Two code paths build the same Lucene string: the driver-ops builder used when
``driver.search_interface`` is wired, and ``search_utils.fulltext_query`` used otherwise
(Neo4j goes down the second one). Both must parenthesise the group filter."""

from types import SimpleNamespace

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neo4j.operations.search_ops import _build_neo4j_fulltext_query
from graphiti_core.search.search_utils import fulltext_query

FAKE_DRIVER = SimpleNamespace(provider=GraphProvider.NEO4J, fulltext_syntax='')


def test_legacy_fulltext_query_parenthesises_the_group_filter():
    assert fulltext_query('NetAlertX', ['minamo', 'haruo'], FAKE_DRIVER) == (
        '(group_id:"minamo" OR group_id:"haruo") AND (\\Net\\AlertX)'
    )


def test_legacy_fulltext_query_without_groups():
    assert fulltext_query('NetAlertX', None, FAKE_DRIVER) == '(\\Net\\AlertX)'


def test_group_filter_is_parenthesised_before_the_query():
    """`a OR b AND (query)` must not leak into Lucene: without the parens the first group
    matches unconditionally and the leg returns query-independent facts."""
    built = _build_neo4j_fulltext_query('NetAlertX', ['minamo', 'haruo'])
    assert built == '(group_id:"minamo" OR group_id:"haruo") AND (\\Net\\AlertX)'


def test_single_group_is_parenthesised():
    built = _build_neo4j_fulltext_query('NetAlertX', ['haruo'])
    assert built == '(group_id:"haruo") AND (\\Net\\AlertX)'


def test_no_group_filter_leaves_the_query_alone():
    assert _build_neo4j_fulltext_query('NetAlertX', None) == '(\\Net\\AlertX)'
    assert _build_neo4j_fulltext_query('NetAlertX', []) == '(\\Net\\AlertX)'


def test_query_is_still_escaped():
    built = _build_neo4j_fulltext_query('a:b (c)', ['g1'])
    assert built.startswith('(group_id:"g1") AND (')
    assert built.endswith(')')
    assert '(' in built[27:]  # the sanitised query keeps its own parens intact
