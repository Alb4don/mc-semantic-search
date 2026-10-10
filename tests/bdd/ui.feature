Feature: Search page (HTML + htmx)
  A fast, simple search page in Portuguese with live results (Story 4.8).

  Scenario: Live search as I type
    Given I am on the search page
    When I type "bolha" in the search box
    Then results appear without a full page reload (htmx)
    And each result shows title, episode, date, channel and snippet

  Scenario: Empty results distinguish states
    When I search for something with no match
    Then the page shows "Nenhum resultado encontrado" with the archive freshness line
    And the page does not look identical to a loading state

  Scenario: Freshness is visible
    Given the archive was last updated 2026-10-09
    When I open the search page
    Then the footer shows "acervo atualizado até 2026-10-09"
