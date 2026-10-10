Feature: Literal-aware ranking boost (Story 4.3)
  Verbatim-phrase searches must surface the exact moment a phrase was
  said — paraphrases are demoted, never dropped — while natural-language
  searches keep on-topic chunks on top: chunks are demoted only by the
  query terms they lack, never dropped.

  Scenario: Verbatim query outranks paraphrase
    Given the archive contains a chunk with the literal phrase "bitcoin é bolha"
      And a semantically similar chunk without that phrase
    When I search for "bitcoin é bolha"
    Then the literal chunk is ranked first
    And its snippet contains the exact phrase

  Scenario: Paraphrase search still finds the idea
    When I search for "mercado em alta mesmo com inflação"
    Then the top results are chunks semantically about that idea
    And results are demoted only by the query terms they lack
