#!/usr/bin/env bash
#
# uts-vpn.sh - connect to the UTS VPN from NixOS, then reach the UTS HPC.
#
# UTS runs F5 BIG-IP APM at vpn.uts.edu.au. The login uses Okta SSO with MFA.
# openconnect cannot drive that login on its own, because the form needs a real
# browser and JavaScript. So you log in with a browser once, copy the MRHSession
# cookie, and this script does every other step.
#
# Usage:
#   ./uts-vpn.sh                     log in with the browser, then paste it
#   ./uts-vpn.sh --cookie VALUE      pass the cookie directly
#   ./uts-vpn.sh --cookie-file PATH  read the cookie from a file
#   ./uts-vpn.sh --clip              read the cookie from the X clipboard
#   ./uts-vpn.sh --help              show this help
#
# UTS_COOKIE=VALUE ./uts-vpn.sh also works. Prefer that or --cookie-file over
# --cookie: argv is world readable through ps for the whole session, but the
# environment of a process is not.

set -euo pipefail

VPN_HOST="${UTS_VPN_HOST:-vpn.uts.edu.au}"
PROTOCOL="f5"
COOKIE_NAME="MRHSession"
PIN_FILE="${XDG_CACHE_HOME:-$HOME/.cache}/uts-vpn/servercert"

USE_CLIP=0
cookie="${UTS_COOKIE:-}"
ORIG_ARGS=( "$@" )

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

usage() { sed -n '3,20p' "$0" | sed 's/^# \?//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --clip) USE_CLIP=1 ;;
    --cookie)      [ $# -ge 2 ] || die "--cookie needs a value";     cookie="$2"; shift ;;
    --cookie=*)    cookie="${1#*=}" ;;
    --cookie-file) [ $# -ge 2 ] || die "--cookie-file needs a path"; COOKIE_FILE="$2"; shift ;;
    --cookie-file=*) COOKIE_FILE="${1#*=}" ;;
    -h|--help) usage ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

if [ -n "${COOKIE_FILE:-}" ]; then
  [ -r "$COOKIE_FILE" ] || die "cannot read $COOKIE_FILE"
  cookie="$(<"$COOKIE_FILE")"
fi

# --- make sure we have a terminal ------------------------------------------
# Both the cookie prompt and sudo need a tty. A pipe, cron, or the Claude Code
# "!" passthrough does not give one. Reopen in a terminal emulator instead of
# failing. This runs before the nix bootstrap, so the new window bootstraps
# itself cleanly.
if [ ! -t 0 ] && [ -z "${UTS_VPN_NO_RELAUNCH:-}" ]; then
  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    for term in alacritty st xterm; do
      command -v "$term" >/dev/null 2>&1 || continue
      info "No terminal attached. Reopening in $term."
      export UTS_VPN_NO_RELAUNCH=1
      # Keep the window open so an error stays readable.
      # shellcheck disable=SC2016  # the inner shell must expand these, not this one
      exec "$term" -e bash -c \
        'set +e; "$0" "$@"; printf "\n[exit %s] Press Enter to close..." "$?"; read -r _' \
        "$(realpath "$0")" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
    done
  fi
  die "no terminal available. Run this from a terminal window."
fi

# --- NixOS: make openconnect available -------------------------------------
# openconnect is not in the system closure by default. If it is missing, re-run
# this script inside a nix shell. The guard variable stops an infinite loop.
if ! command -v openconnect >/dev/null 2>&1; then
  if [ -n "${UTS_VPN_BOOTSTRAPPED:-}" ]; then
    die "nix shell did not provide openconnect"
  fi
  command -v nix >/dev/null 2>&1 || die "openconnect is missing and nix is not available"
  info "openconnect not installed. Starting a nix shell."
  export UTS_VPN_BOOTSTRAPPED=1
  exec nix shell nixpkgs#openconnect nixpkgs#vpnc-scripts --command "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}
fi

# sudo resets PATH, so a store path from `nix shell` disappears. Resolve the
# absolute path now and hand that to sudo.
OPENCONNECT="$(command -v openconnect)"

# --- NixOS: locate vpnc-script ---------------------------------------------
# There is no /etc/vpnc/vpnc-script on NixOS. The nixpkgs openconnect wrapper
# normally sets this itself, but pass it explicitly so the script also works
# when openconnect comes from `nix shell`.
VPNC_SCRIPT=""
for candidate in \
  "${VPNC_SCRIPT_OVERRIDE:-}" \
  "$(command -v vpnc-script 2>/dev/null || true)" \
  /etc/vpnc/vpnc-script
do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then VPNC_SCRIPT="$candidate"; break; fi
done
if [ -z "$VPNC_SCRIPT" ] && command -v nix >/dev/null 2>&1; then
  VPNC_SCRIPT="$(nix build --no-link --print-out-paths nixpkgs#vpnc-scripts 2>/dev/null || true)/bin/vpnc-script"
  [ -x "$VPNC_SCRIPT" ] || VPNC_SCRIPT=""
fi
[ -n "$VPNC_SCRIPT" ] || warn "no vpnc-script found. Routing and DNS may not be set."

# --- step 1: browser login --------------------------------------------------
if [ -z "$cookie" ]; then
cat <<EOF

  1. A browser window opens at https://$VPN_HOST/
  2. Log in with your UTS account and complete MFA.
  3. Ignore the "Waiting for Network Access Application" box. It waits for a
     client that does not exist on Linux. Leave the tab open.
  4. Open the developer tools with F12, then find the cookie list:
       Firefox:  Storage      -> Cookies -> https://$VPN_HOST
       Chrome:   Application  -> Cookies -> https://$VPN_HOST
     In Firefox the Storage tab may hide behind the ">>" chevron.
  5. Copy the value of the "$COOKIE_NAME" cookie.

EOF

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "https://$VPN_HOST/" >/dev/null 2>&1 &
  disown || true
else
  warn "xdg-open is missing. Open https://$VPN_HOST/ yourself."
fi
fi

# --- step 2: collect the cookie --------------------------------------------
if [ -n "$cookie" ]; then
  :  # already supplied by --cookie, --cookie-file or $UTS_COOKIE
elif [ "$USE_CLIP" -eq 1 ]; then
  command -v xclip >/dev/null 2>&1 || die "--clip needs xclip"
  read -r -p "Press Enter once the cookie is on the clipboard... " _
  cookie="$(xclip -selection clipboard -o 2>/dev/null || true)"
  [ -n "$cookie" ] || die "the clipboard is empty"
else
  # -s keeps the session token off the screen.
  read -r -s -p "Paste the $COOKIE_NAME cookie: " cookie
  echo
fi

# Accept a bare value or a full "MRHSession=..." pair, with or without quotes.
cookie="${cookie//[$'\t\r\n ']/}"
cookie="${cookie#\"}"; cookie="${cookie%\"}"
cookie="${cookie#\'}"; cookie="${cookie%\'}"
cookie="${cookie#"${COOKIE_NAME}"=}"

[ -n "$cookie" ] || die "no cookie given"
if ! [[ "$cookie" =~ ^[A-Za-z0-9]{16,}$ ]]; then
  warn "that does not look like an $COOKIE_NAME value. Trying it anyway."
fi

# --- step 3: connect --------------------------------------------------------
# Ask for the sudo password now. openconnect takes the cookie on stdin, so sudo
# cannot read a password there later.
info "Asking for sudo."
sudo -v

oc_args=( --protocol="$PROTOCOL" --cookie-on-stdin )
[ -n "${UTS_USER:-}" ]  && oc_args+=( --user="$UTS_USER" )
[ -n "$VPNC_SCRIPT" ]   && oc_args+=( --script="$VPNC_SCRIPT" )

# Pin the server certificate after the first connection.
if [ -s "$PIN_FILE" ]; then
  oc_args+=( --servercert="$(cat "$PIN_FILE")" )
else
  mkdir -p "$(dirname "$PIN_FILE")"
  warn "no pinned certificate yet. Check the fingerprint and save it to:"
  warn "  $PIN_FILE"
fi

info "Connecting to $VPN_HOST over $PROTOCOL."
info "Press Ctrl-C to disconnect."

# The cookie goes in on stdin, never in argv, so it stays out of `ps`.
printf '%s=%s\n' "$COOKIE_NAME" "$cookie" \
  | sudo -n "$OPENCONNECT" "${oc_args[@]}" "$VPN_HOST"
