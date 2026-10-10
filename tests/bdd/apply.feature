Feature: Apply queued increments (Mode B server)
  `mc apply` consumes the increment files pushed to the server queue
  directory (Story 3.7): embed + upsert each video once, tracked per
  chunk content hash so a re-run is a no-op; corrupt files are
  quarantined instead of blocking the rest of the queue.

  Scenario: Increment applied exactly once
    Given a queued increment with 3 chunks
    When "mc apply" runs twice
    Then the second run is a no-op (chunks tracked by content hash)
    And all 3 chunks are searchable

  Scenario: Corrupt increment is quarantined
    Given a malformed JSONL file in the queue directory
    And another queued file with 2 valid chunks
    When "mc apply" runs
    Then the file is moved to queue/failed/ with the error logged
    And other queued files are still applied
