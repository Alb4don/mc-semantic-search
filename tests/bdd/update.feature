Feature: Daily `mc update` ingestion
  Scrape → download → chunk → embed → upsert → mark processed, so the
  index gains new episodes unattended (Story 3.4).

  Scenario: New video is ingested
    Given channels return video IDs ["new1"] and none processed
    When `mc update` runs successfully
    Then "new1" is searchable in Qdrant with 3 chunk points
    And the state store marks "new1" processed after the upsert commits
    And the queue staging file contains the 3 chunk records

  Scenario: Already-processed videos are skipped
    Given channels return ["old1", "old2"] and both are marked processed
    When `mc update` runs
    Then no transcript download happens for "old1" or "old2"
    And the run log reports "0 new videos"

  Scenario: Download failure does not lose the rest
    Given video "bad1" fails after retries and "good1" succeeds
    When `mc update` finishes
    Then "good1" is indexed, "bad1" is in the ignore list
    And the run exits non-zero with a summary of the failure
