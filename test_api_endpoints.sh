#!/bin/bash

# Test script for D&D DM Copilot API endpoints
# Based on plan from ~/.claude/plans/zany-humming-garden.md

set -e

BASE_URL="http://localhost:8000"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "D&D DM Copilot API Endpoint Tests"
echo "========================================="
echo ""

# Function to print test results
print_result() {
    local endpoint=$1
    local status=$2
    local response=$3

    if [ "$status" = "200" ] || [ "$status" = "201" ]; then
        echo -e "${GREEN}✓ PASS${NC} - $endpoint (HTTP $status)"
        echo "Response: $response"
    else
        echo -e "${RED}✗ FAIL${NC} - $endpoint (HTTP $status)"
        echo "Response: $response"
    fi
    echo ""
}

# Test 1: Health Check (GET /health)
echo "Test 1: Health Check"
echo "GET $BASE_URL/health"
response=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
print_result "GET /health" "$http_code" "$body"

# Test 2: Mechanics Query (POST /api/v1/mechanics/query)
echo "Test 2: Mechanics Query - Basic"
echo "POST $BASE_URL/api/v1/mechanics/query"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/mechanics/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "How does Divine Smite work?", "top_k": 3}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
print_result "POST /api/v1/mechanics/query" "$http_code" "$body"

# Test 3: Mechanics Query - Different top_k
echo "Test 3: Mechanics Query - top_k=5"
echo "POST $BASE_URL/api/v1/mechanics/query"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/mechanics/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What are the rules for flanking?", "top_k": 5}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
print_result "POST /api/v1/mechanics/query (top_k=5)" "$http_code" "$body"

# Test 4: Mechanics Query - Edge case (empty query should fail validation)
echo "Test 4: Mechanics Query - Empty query (should fail)"
echo "POST $BASE_URL/api/v1/mechanics/query"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/mechanics/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "", "top_k": 3}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
if [ "$http_code" = "422" ]; then
    echo -e "${GREEN}✓ PASS${NC} - Empty query validation (HTTP $http_code - expected failure)"
else
    echo -e "${RED}✗ FAIL${NC} - Empty query validation (HTTP $http_code - should be 422)"
fi
echo "Response: $body"
echo ""

# Test 5: Mechanics Query - Edge case (invalid top_k should fail validation)
echo "Test 5: Mechanics Query - Invalid top_k=0 (should fail)"
echo "POST $BASE_URL/api/v1/mechanics/query"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/mechanics/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "top_k": 0}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
if [ "$http_code" = "422" ]; then
    echo -e "${GREEN}✓ PASS${NC} - Invalid top_k validation (HTTP $http_code - expected failure)"
else
    echo -e "${RED}✗ FAIL${NC} - Invalid top_k validation (HTTP $http_code - should be 422)"
fi
echo "Response: $body"
echo ""

# Test 6: Mechanics Query - Complex D&D question
echo "Test 6: Mechanics Query - Complex question"
echo "POST $BASE_URL/api/v1/mechanics/query"
response=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/api/v1/mechanics/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Can a wizard cast a spell as a bonus action and then cast another spell as an action on the same turn?", "top_k": 3}')
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')
print_result "POST /api/v1/mechanics/query (complex)" "$http_code" "$body"

echo "========================================="
echo "Summary"
echo "========================================="
echo -e "${YELLOW}Note: Endpoints planned but not yet implemented:${NC}"
echo "- POST /api/v1/lore/query"
echo "- POST /api/v1/lore/load"
echo "- POST /api/v1/query (combined)"
echo ""
echo "Tests completed!"
