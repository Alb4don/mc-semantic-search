Feature: Embedded player at the exact moment
  Each result embeds the YouTube player (youtube-nocookie) starting at the
  chunk timestamp, click-to-load only (Story 4.9).

  Scenario: Player loads the exact moment on demand
    Given a result in the archive with a chunk at 1534 seconds
    When I open the result's player placeholder
    Then the iframe points at youtube-nocookie with start=1534
    And the iframe is lazy-loaded
    And the player is not rendered until user action (click-to-load)
