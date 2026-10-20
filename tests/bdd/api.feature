Feature: GET /search JSON API (Story 4.5)

  The API serves hybrid search results as JSON, with server-side limits,
  a freshness block and a clean 503 when Qdrant is unreachable.

  Scenario: Searching returns contract-shaped results with snippets
    Given a seeded archive of 55 transcript chunks
    When I search for "bitcoin"
    Then the response is 200
    And the body matches the /search contract
    And every result snippet highlights the query word
    And exactly 10 results are returned

  Scenario: A limit above 50 is clamped server-side
    Given a seeded archive of 55 transcript chunks
    When I search with query "bitcoin" and limit 200
    Then the response is 200
    And exactly 50 results are returned

  Scenario: An empty archive still answers with freshness
    Given an empty collection
    When I search for "bitcoin"
    Then the response is 200
    And the results list is empty
    And the freshness block is present

  Scenario: Qdrant down yields 503, not 500
    Given the search service has no reachable qdrant
    When I search for "bitcoin"
    Then the response is 503

  Scenario: The landing page answers in JSON when asked
    Given a seeded archive of 55 transcript chunks
    When I ask the landing page for JSON with query "bitcoin"
    Then the response is 200
    And the body matches the render=json envelope
    And every result snippet highlights the query word

  Scenario: Muriel introduces herself to agents
    Given a seeded archive of 55 transcript chunks
    When I open the landing page as JSON without a query
    Then the response is 200
    And the agent card describes the archive and its endpoints
