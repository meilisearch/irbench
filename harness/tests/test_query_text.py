from irbench.terms import plain_query_text


def test_operator_characters_are_removed_for_every_engine():
    # `"` (phrase) and a leading `-` (exclusion) are operators in Meilisearch, Typesense and
    # Postgres websearch_to_tsquery but plain text in the others: queries are sent as plain text.
    assert plain_query_text('fawkes 36" blue vanity') == "fawkes 36 blue vanity"
    assert plain_query_text('Is hidden content -indexed- by "search engines"?') == (
        "Is hidden content indexed- by search engines ?"
    )
    assert plain_query_text("--force flag") == "force flag"


def test_hyphens_inside_words_are_kept():
    assert plain_query_text("covid-19 long-term effects") == "covid-19 long-term effects"
    assert plain_query_text("  a  -  b ") == "a b"
