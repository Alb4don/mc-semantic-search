Feature: Search page (HTML + htmx)
  A fast, simple search page in Portuguese with live results (Story 4.8).

  Scenario: Live search as I type
    Given I am on the search page
    When I type "bolha" in the search box
    Then results appear without a full page reload (htmx)
    And each result shows title, episode, date, channel and snippet

  Scenario: Empty results distinguish states
    When I search for something with no match
    Then the page shows "Muriel folheou o acervo inteiro" with the archive freshness line
    And the page does not look identical to a loading state

  Scenario: Freshness is visible
    Given the archive was last updated 2026-10-09
    When I open the search page
    Then the footer shows "acervo atualizado até 2026-10-09"

  Scenario: The page teaches agents how to use it
    Given I am on the search page
    Then the page shows how to point an agent at Muriel

  Scenario: Archive status is visible below everything
    Given the archive has 3556 catalogued videos totalling 197342 seconds
    And the last episode loaded is #971 from 2026-10-20
    When I open the search page
    Then the status block is the last content inside the footer
    And it shows "último episódio" "#971 · 2026-10-20"
    And it shows "vídeos catalogados" "3.556"
    And it shows "duração total" "197.342 segundos · 3.289,03 minutos · 54,82 horas · 2,28 dias"

  Scenario: Status hides when the archive reports nothing
    Given the archive has no recorded stats
    When I open the search page
    Then the page renders without the status block
    And the freshness line and colophon notes still show

  Scenario: A broken stats store never breaks the page
    Given the state store raises on stats reads
    When I open the search page
    Then the page still returns 200 with the search console and agent note intact
