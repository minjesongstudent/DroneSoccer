#!/bin/bash
# Sourced by the Finder launchers; no network or drone access here.
cd "$(dirname "$0")" || exit 1
pause_window() {
  printf '\nPress Return to close this window. '
  read -r ignored
}
find_python() {
  PYTHON_BIN=''
  for candidate in /Library/Frameworks/Python.framework/Versions/Current/bin/python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3 2>/dev/null)"; do
    # Avoid Apple's developer-tools stub triggering an unrelated Xcode install.
    [ "$candidate" = /usr/bin/python3 ] && continue
    if [ -x "$candidate" ] && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      return 0
    fi
  done
  echo 'Python 3.10 or newer is needed. Install the macOS universal2 installer from python.org.'
  echo 'After installation, run 1-Setup.command again. No GPU or pip packages are needed.'
  open 'https://www.python.org/downloads/macos/'
  return 1
}
