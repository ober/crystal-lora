#!/bin/bash
# Manage RunPod serverless endpoints for the Crystal Qwen model.
#
# Commands:
#   list              List all serverless endpoints
#   health [ID]       Check endpoint health (workers, queue depth)
#   test [ID]         Send a test prompt to an endpoint
#   delete [ID]       Delete a specific endpoint (stops billing)
#   delete-all        Delete all crystal-qwen endpoints
#   purge [ID]        Scale workers to 0 to clear stale jobs
#   restore [ID]      Restore workers to 0-1 after purge
#
# Requires: RUNPOD_API_KEY env var
#
# Usage:
#   ./manage_runpod.sh list
#   ./manage_runpod.sh health abc123
#   ./manage_runpod.sh delete abc123

set -euo pipefail

if [ "${1:-}" = "help" ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ] || [ -z "${1:-}" ]; then
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  list              List all serverless endpoints"
    echo "  health <ID>       Check endpoint health (workers, queue)"
    echo "  test <ID>         Send a test prompt"
    echo "  delete <ID>       Delete a specific endpoint (stops billing)"
    echo "  delete-all        Delete all crystal-qwen endpoints"
    echo "  purge <ID>        Scale workers to 0 (clear stale jobs)"
    echo "  restore <ID>      Restore workers to 0-1 after purge"
    echo ""
    echo "Requires: RUNPOD_API_KEY env var"
    exit 0
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set."
    echo "Get your key from: https://www.runpod.io/console/user/settings"
    echo "Then run: export RUNPOD_API_KEY=your-key"
    exit 1
fi

RUNPOD_URL="https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}"

runpod_graphql() {
    local query="$1"
    curl -s --request POST \
        --header 'content-type: application/json' \
        --url "$RUNPOD_URL" \
        --data "{\"query\": \"$query\"}"
}

cmd_list() {
    echo "=== RunPod Serverless Endpoints ==="
    echo ""

    local result
    result=$(runpod_graphql "{ myself { serverlessDiscount endpoints { id name templateId gpuIds workersMin workersMax idleTimeout } } }")

    if ! echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
endpoints = data['data']['myself']['endpoints']
if not endpoints:
    print('No endpoints found.')
    sys.exit(0)
for ep in endpoints:
    print(f\"  {ep['id']}  {ep['name']:<25} GPU: {ep['gpuIds']:<12} Workers: {ep['workersMin']}-{ep['workersMax']}  Idle: {ep['idleTimeout']}s\")
print(f\"\n  Total: {len(endpoints)} endpoint(s)\")
" 2>/dev/null; then
        echo "ERROR: Failed to list endpoints."
        echo "Response: $result"
        exit 1
    fi
}

cmd_health() {
    local endpoint_id="${1:-}"
    if [ -z "$endpoint_id" ]; then
        echo "Usage: $0 health <ENDPOINT_ID>"
        exit 1
    fi

    echo "=== Endpoint Health: $endpoint_id ==="
    echo ""

    local health
    health=$(curl -s \
        --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
        "https://api.runpod.ai/v2/${endpoint_id}/health")

    echo "$health" | python3 -c "
import sys, json
data = json.load(sys.stdin)
workers = data.get('workers', {})
print(f\"  Ready workers:      {workers.get('ready', 0)}\")
print(f\"  Running workers:    {workers.get('running', 0)}\")
print(f\"  Throttled workers:  {workers.get('throttled', 0)}\")
print(f\"  Initializing:       {workers.get('initializing', 0)}\")
print()
jobs = data.get('jobs', {})
print(f\"  Jobs completed:     {jobs.get('completed', 0)}\")
print(f\"  Jobs failed:        {jobs.get('failed', 0)}\")
print(f\"  Jobs in progress:   {jobs.get('inProgress', 0)}\")
print(f\"  Jobs in queue:      {jobs.get('inQueue', 0)}\")
print(f\"  Jobs retried:       {jobs.get('retried', 0)}\")
" 2>/dev/null || {
        echo "ERROR: Failed to get health."
        echo "Response: $health"
        exit 1
    }
}

cmd_test() {
    local endpoint_id="${1:-}"
    if [ -z "$endpoint_id" ]; then
        echo "Usage: $0 test <ENDPOINT_ID>"
        exit 1
    fi

    # Auto-detect HF_REPO from .together_state.json or fall back
    local hf_repo
    hf_repo=$(python3 -c "
import os, json
d = '.together_state.json'
if os.path.exists(d):
    s = json.load(open(d))
    print(s.get('hf_repo','jaimef21/crystal-qwen-30b'))
else:
    print('jaimef21/crystal-qwen-30b')
" 2>/dev/null || echo "jaimef21/crystal-qwen-30b")

    echo "=== Test Prompt: $endpoint_id ==="
    echo "Model: $hf_repo"
    echo "Sending: 'How do I declare a class with type-annotated properties in Crystal?'"
    echo ""

    local result
    result=$(curl -s \
        --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
        --header "Content-Type: application/json" \
        "https://api.runpod.ai/v2/${endpoint_id}/openai/v1/chat/completions" \
        --data "{
            \"model\": \"$hf_repo\",
            \"messages\": [{\"role\": \"user\", \"content\": \"How do I declare a class with type-annotated properties in Crystal?\"}],
            \"max_tokens\": 256,
            \"temperature\": 0.7
        }")

    echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'choices' in data:
    print(data['choices'][0]['message']['content'])
elif 'error' in data:
    print(f\"Error: {data['error']}\")
else:
    print(json.dumps(data, indent=2))
" 2>/dev/null || {
        echo "Response: $result"
    }
}

cmd_delete() {
    local endpoint_id="${1:-}"
    if [ -z "$endpoint_id" ]; then
        echo "Usage: $0 delete <ENDPOINT_ID>"
        exit 1
    fi

    echo "Deleting endpoint: $endpoint_id ..."

    local result
    result=$(runpod_graphql "mutation { deleteEndpoint(id: \\\"$endpoint_id\\\") }")

    if echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('data', {}).get('deleteEndpoint') is not None:
    sys.exit(0)
sys.exit(1)
" 2>/dev/null; then
        echo "Deleted endpoint $endpoint_id"
        echo ""
        echo "Billing for this endpoint has stopped."
    else
        echo "ERROR: Failed to delete endpoint."
        echo "Response: $result"
        exit 1
    fi
}

cmd_delete_all() {
    echo "=== Finding crystal-qwen endpoints to delete ==="
    echo ""

    local result
    result=$(runpod_graphql "{ myself { endpoints { id name } } }")

    local ids
    ids=$(echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
endpoints = data['data']['myself']['endpoints']
matches = [ep for ep in endpoints if 'crystal' in ep['name'].lower()]
if not matches:
    print('')
    sys.exit(0)
for ep in matches:
    print(ep['id'])
" 2>/dev/null)

    if [ -z "$ids" ]; then
        echo "No crystal-qwen endpoints found."
        return
    fi

    echo "Found endpoints to delete:"
    echo "$ids" | while read -r id; do
        echo "  $id"
    done
    echo ""
    read -rp "Delete all? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Cancelled."
        return
    fi

    echo ""
    echo "$ids" | while read -r id; do
        cmd_delete "$id"
    done
}

cmd_purge() {
    local endpoint_id="${1:-}"
    if [ -z "$endpoint_id" ]; then
        echo "Usage: $0 purge <ENDPOINT_ID>"
        exit 1
    fi

    echo "=== Purging endpoint: $endpoint_id ==="
    echo "Scaling workers to 0 to clear stale jobs ..."

    local result
    result=$(runpod_graphql "mutation { saveEndpoint(input: { id: \\\"$endpoint_id\\\", workersMin: 0, workersMax: 0 }) { id workersMin workersMax } }")

    if echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ep = data['data']['saveEndpoint']
print(f\"  Workers scaled to {ep['workersMin']}-{ep['workersMax']}\")
" 2>/dev/null; then
        echo ""
        echo "Workers scaled to 0. Stale jobs will be cleared."
        echo ""
        echo "To restore, run:"
        echo "  $0 restore $endpoint_id"
    else
        echo "ERROR: Failed to purge endpoint."
        echo "Response: $result"
        exit 1
    fi
}

cmd_restore() {
    local endpoint_id="${1:-}"
    if [ -z "$endpoint_id" ]; then
        echo "Usage: $0 restore <ENDPOINT_ID>"
        exit 1
    fi

    echo "Restoring endpoint: $endpoint_id (workers: 0-1) ..."

    local result
    result=$(runpod_graphql "mutation { saveEndpoint(input: { id: \\\"$endpoint_id\\\", workersMin: 0, workersMax: 1 }) { id workersMin workersMax } }")

    if echo "$result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
ep = data['data']['saveEndpoint']
print(f\"  Workers restored to {ep['workersMin']}-{ep['workersMax']}\")
" 2>/dev/null; then
        echo "Done."
    else
        echo "ERROR: Failed to restore endpoint."
        echo "Response: $result"
        exit 1
    fi
}

cmd_help() {
    echo "Usage: $0 <command> [args]"
    echo ""
    echo "Commands:"
    echo "  list              List all serverless endpoints"
    echo "  health <ID>       Check endpoint health (workers, queue)"
    echo "  test <ID>         Send a test prompt"
    echo "  delete <ID>       Delete a specific endpoint (stops billing)"
    echo "  delete-all        Delete all crystal-qwen endpoints"
    echo "  purge <ID>        Scale workers to 0 (clear stale jobs)"
    echo "  restore <ID>      Restore workers to 0-1 after purge"
    echo ""
    echo "Requires: RUNPOD_API_KEY env var"
}

case "${1:-help}" in
    list)       cmd_list ;;
    health)     cmd_health "${2:-}" ;;
    test)       cmd_test "${2:-}" ;;
    delete)     cmd_delete "${2:-}" ;;
    delete-all) cmd_delete_all ;;
    purge)      cmd_purge "${2:-}" ;;
    restore)    cmd_restore "${2:-}" ;;
    help|--help|-h) cmd_help ;;
    *)
        echo "Unknown command: $1"
        echo ""
        cmd_help
        exit 1
        ;;
esac
