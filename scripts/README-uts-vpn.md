# UTS VPN on NixOS

`uts-vpn.sh` connects this machine to the UTS VPN, so you can reach the UTS HPC
from home.

## What UTS runs

UTS runs two separate remote access systems.

| Host | System | Works on Linux |
|---|---|---|
| `vpn.uts.edu.au` | F5 BIG-IP APM | Yes, through openconnect |
| `studyaccess.uts.edu.au` | Akamai Zero Trust | No |

Use `vpn.uts.edu.au`. The Akamai Zero Trust Client supports only macOS and
Windows. The older Akamai EAA client supported Ubuntu, but it reached end of
life in August 2026.

The F5 login uses Okta SSO with MFA. openconnect cannot drive that login,
because the form needs a real browser and JavaScript. So you log in once with a
browser and copy the `MRHSession` cookie. This is the one manual step.

## Use

1. Run `./uts-vpn.sh`.
2. Log in in the browser window that opens.
3. Open the developer tools with F12.
4. Ignore the "Waiting for Network Access Application" box. It waits for the F5
   Edge Client, which has no Linux build. Leave the tab open.
5. Go to Storage, then Cookies, then `https://vpn.uts.edu.au`.
6. Copy the value of the `MRHSession` cookie.
7. Paste it at the prompt.

### Passing the cookie without the prompt

```
./uts-vpn.sh --cookie <value>          # argv
./uts-vpn.sh --cookie-file ./ck.txt    # file
./uts-vpn.sh --clip                    # X clipboard
UTS_COOKIE=<value> ./uts-vpn.sh        # environment
```

When the cookie arrives by any of these, the script skips the browser step and
connects straight away.

Prefer `--cookie-file` or `UTS_COOKIE` over `--cookie`. Linux makes
`/proc/<pid>/cmdline` readable by every user, so a cookie in argv stays visible
in `ps` for the whole session. The environment of a process is readable only by
its owner.

### Why the cookie cannot come from the cookie database

`MRHSession` is a session cookie. Firefox holds session cookies in memory and
never writes them to `cookies.sqlite`. So no script can read it from disk. The
copy step stays manual.

In Firefox the cookie list sits under **Storage**, not Application. The tab may
hide behind the `>>` chevron.

## Is it connected?

openconnect holds the tunnel in the foreground. It does not exit and it prints
nothing more once it settles. The success line is:

```
Configured as 172.28.34.140, with SSL connected and DTLS disabled
```

Check the state from another terminal:

```
ip -br addr show tun0          # expect an address such as 172.28.34.140/32
getent hosts adsroot.uts.edu.au  # expect 10.242.1.8, an internal address
```

Press Ctrl-C in the openconnect window to disconnect.

The server sets an idle timeout of 15 minutes.

The "no pinned certificate" warning is harmless. UTS presents a valid
certificate, so openconnect accepts it without a pin.

## NixOS notes

The script handles five NixOS problems.

1. **openconnect is not installed.** The script starts a `nix shell` when the
   binary is missing. To install it properly, add `openconnect` to
   `environment.systemPackages` in `myNixSystems/nixos/packages.nix`, then run
   `sudo nixos-rebuild switch`.
2. **sudo clears PATH.** A binary from `nix shell` lives in the store and
   disappears from the sudo environment. The script resolves the absolute store
   path first, then passes that path to sudo.
3. **There is no `/etc/vpnc/vpnc-script`.** The script finds `vpnc-script` on
   PATH, or builds `nixpkgs#vpnc-scripts` and uses that path.
4. **There may be no tty.** A pipe or the Claude Code `!` passthrough gives no
   terminal, so the prompt and sudo both fail. The script reopens itself in
   `alacritty`, `st` or `xterm` and holds the window open on error.
5. **sudo and stdin conflict.** openconnect reads the cookie on stdin, so sudo
   cannot read a password there. The script calls `sudo -v` first to cache the
   credential.

DNS needs no extra work on this machine. `systemd-resolved` is inactive and
`resolvconf` writes `/etc/resolv.conf`, which the standard `vpnc-script`
already supports.

The script calls `nixpkgs#...`, which points at nixpkgs-unstable in your
registry. Your system flake pins nixos-26.05. Installing openconnect
declaratively removes this mismatch.

## Security

The cookie is a live session token. The script passes it to openconnect on
stdin with `--cookie-on-stdin`, never in the command line, so it does not show
up in `ps`.

After the first connection, save the server certificate fingerprint to
`~/.cache/uts-vpn/servercert`. The script then pins it with `--servercert`.

## After connecting

SSH to the HPC login node in a second terminal.

```
ssh u<your-student-id>@<login-node-ip>
```

eResearch emails you the IP address of the login node. Set `UTS_USER` to skip
the openconnect username prompt, if it asks for one.
