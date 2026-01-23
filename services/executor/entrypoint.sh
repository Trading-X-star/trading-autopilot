#!/bin/sh
# Load token from secret file into env (runs as root)
if [ -f "$TINKOFF_TOKEN_FILE" ]; then
    export TINKOFF_TOKEN=$(cat "$TINKOFF_TOKEN_FILE")
fi
# Drop privileges and run app
exec su-exec appuser python main.py
