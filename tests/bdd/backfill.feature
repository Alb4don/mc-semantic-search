Feature: Backfill resumability and completion
  Backfill embeds + indexes the legacy archive with checkpoints so a
  crash resumes instead of restarting (NFR-3).

  Scenario: Backfill resumes after interruption
    Given a legacy DB with 12 chunks across 2 videos and a batch size of 5
    When backfill processes 7 chunks and is killed
    And backfill is started again
    Then the remaining 5 chunks are embedded and indexed
    And no chunk is embedded twice (total embedding calls: 12)

  Scenario: Backfill completes and verifies counts
    Given the legacy DB with 12 chunks in 1 video is fully imported
    When backfill finishes
    Then the Qdrant collection contains 12 points
    And the state store reports 1 video processed
    And a final report prints indexed/quarantine/skipped counts
