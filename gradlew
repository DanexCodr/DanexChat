#!/bin/sh
# Gradle wrapper script (delegates to system Gradle or downloaded wrapper)
set -e
SCRIPT_DIR=$(dirname "$0")

# Use local wrapper jar if available, otherwise fall back to system gradle
if [ -f "$SCRIPT_DIR/gradle/wrapper/gradle-wrapper.jar" ]; then
  exec java -jar "$SCRIPT_DIR/gradle/wrapper/gradle-wrapper.jar" "$@"
else
  exec gradle "$@"
fi
