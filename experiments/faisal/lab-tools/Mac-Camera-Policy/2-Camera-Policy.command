#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo 'Connect to Tello Wi-Fi, keep motors OFF, close all other drone apps.'
read -r -p 'Press Return to start camera/policy preview, or Ctrl+C to cancel. ' ignored
.venv/bin/python camera_policy.py
read -r -p 'Press Return to close. ' ignored
