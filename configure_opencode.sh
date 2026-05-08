#!/bin/bash
# Configure OpenCode to use the Crystal Qwen model.
#
# Commands:
#   ollama                 Configure for local Ollama
#   runpod <ENDPOINT_ID>   Configure for RunPod endpoint
#   both <ENDPOINT_ID>     Configure both Ollama and RunPod
#
# Writes to ~/.config/opencode/opencode.json, preserving existing MCP config.
#
# Requires: RUNPOD_API_KEY env var (for runpod/both modes)
#
# Usage:
#   ./configure_opencode.sh ollama
#   ./configure_opencode.sh runpod abc123
#   ./configure_opencode.sh both abc123

set -euo pipefail

CONFIG_DIR="$HOME/.config/opencode"
CONFIG_FILE="$CONFIG_DIR/opencode.json"

cmd_help() {
    echo "Usage: $0 <mode> [args]"
    echo ""
    echo "Modes:"
    echo "  ollama                 Configure for local Ollama"
    echo "  runpod <ENDPOINT_ID>   Configure for RunPod serverless"
    echo "  both <ENDPOINT_ID>     Configure both providers"
    echo ""
    echo "Writes to: $CONFIG_FILE"
}

write_config() {
    local providers_json="$1"
    local existing_mcp="{}"

    if [ -f "$CONFIG_FILE" ]; then
        existing_mcp=$(python3 -c "
import json, sys
try:
    with open('$CONFIG_FILE') as f:
        data = json.load(f)
    mcp = data.get('mcp', {})
    if mcp:
        print(json.dumps(mcp))
    else:
        print('{}')
except:
    print('{}')
" 2>/dev/null || echo "{}")
    fi

    mkdir -p "$CONFIG_DIR"

    python3 -c "
import json

providers = json.loads('''$providers_json''')
mcp = json.loads('''$existing_mcp''')

config = {
    '\$schema': 'https://opencode.ai/config.json',
    'provider': providers
}

if mcp and mcp != {}:
    config['mcp'] = mcp

with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)
    f.write('\n')

print(json.dumps(config, indent=2))
"
}

ollama_provider() {
    cat <<'EOF'
{
    "ollama": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ollama (local)",
        "options": {
            "baseURL": "http://localhost:11434/v1"
        },
        "models": {
            "crystal-qwen": {
                "name": "Crystal Qwen"
            }
        }
    }
}
EOF
}

runpod_provider() {
    local endpoint_id="$1"
    local api_key="$2"
    cat <<EOF
{
    "runpod": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "RunPod (serverless)",
        "options": {
            "baseURL": "https://api.runpod.ai/v2/${endpoint_id}/openai/v1",
            "apiKey": "${api_key}"
        },
        "models": {
            "crystal-qwen": {
                "name": "Crystal Qwen 30B"
            }
        }
    }
}
EOF
}

both_providers() {
    local endpoint_id="$1"
    local api_key="$2"
    cat <<EOF
{
    "ollama": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ollama (local)",
        "options": {
            "baseURL": "http://localhost:11434/v1"
        },
        "models": {
            "crystal-qwen": {
                "name": "Crystal Qwen"
            }
        }
    },
    "runpod": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "RunPod (serverless)",
        "options": {
            "baseURL": "https://api.runpod.ai/v2/${endpoint_id}/openai/v1",
            "apiKey": "${api_key}"
        },
        "models": {
            "crystal-qwen": {
                "name": "Crystal Qwen 30B"
            }
        }
    }
}
EOF
}

case "${1:-help}" in
    ollama)
        echo "=== Configuring OpenCode for local Ollama ==="
        echo ""
        providers=$(ollama_provider)
        write_config "$providers"
        echo ""
        echo "Written to: $CONFIG_FILE"
        echo ""
        echo "Make sure Ollama is running: ollama serve"
        ;;
    runpod)
        endpoint_id="${2:-}"
        if [ -z "$endpoint_id" ]; then
            echo "Usage: $0 runpod <ENDPOINT_ID>"
            exit 1
        fi
        if [ -z "${RUNPOD_API_KEY:-}" ]; then
            echo "ERROR: RUNPOD_API_KEY not set."
            exit 1
        fi
        echo "=== Configuring OpenCode for RunPod ==="
        echo ""
        providers=$(runpod_provider "$endpoint_id" "$RUNPOD_API_KEY")
        write_config "$providers"
        echo ""
        echo "Written to: $CONFIG_FILE"
        ;;
    both)
        endpoint_id="${2:-}"
        if [ -z "$endpoint_id" ]; then
            echo "Usage: $0 both <ENDPOINT_ID>"
            exit 1
        fi
        if [ -z "${RUNPOD_API_KEY:-}" ]; then
            echo "ERROR: RUNPOD_API_KEY not set."
            exit 1
        fi
        echo "=== Configuring OpenCode for Ollama + RunPod ==="
        echo ""
        providers=$(both_providers "$endpoint_id" "$RUNPOD_API_KEY")
        write_config "$providers"
        echo ""
        echo "Written to: $CONFIG_FILE"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        echo "Unknown mode: $1"
        echo ""
        cmd_help
        exit 1
        ;;
esac
