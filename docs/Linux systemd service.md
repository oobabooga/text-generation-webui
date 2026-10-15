# Run the web UI as a systemd user service

On Linux distributions that use systemd, a user service can start the web UI when you log in and keep its logs in the journal. This example uses the [manual portable installation with venv](https://github.com/oobabooga/textgen#manual-portable-install-with-venv), installed in `~/textgen`.

Complete the installation and start the server successfully from a terminal first. Stop that instance before starting the service so that the ports are free. Install dependencies and download models interactively, not from the service.

## Create the service

Create `~/.config/systemd/user/textgen.service` (create the directory if needed) with:

```ini
[Unit]
Description=Text generation web UI

[Service]
Type=simple
WorkingDirectory=%h/textgen
ExecStart=%h/textgen/venv/bin/python server.py --portable --api --no-electron
Environment=PYTHONNOUSERSITE=1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

`%h` is expanded by systemd to your home directory. Change both paths if your installation is elsewhere. Use the Python executable in the environment where you installed the dependencies; activating the venv in a shell is not necessary. For an extracted portable build, point `WorkingDirectory` at the directory containing `server.py` and `ExecStart` at its `portable_env/bin/python3`.

The command runs the web server without an Electron window or automatic browser launch. Remove `--auto-launch` from `user_data/CMD_FLAGS.txt` if you previously added it there. Model selection and other server flags can be set in that file, or appended to `ExecStart`. For example, add `--model your-model.gguf` to load a downloaded model at startup. Use the same model name that works in your interactive launch.

The working directory matters: relative model, settings, and user-data paths must resolve inside your installation. The service uses your account, including its existing GPU permissions. If your GPU setup needs environment variables, add them with `Environment=NAME=value` lines; systemd does not run your interactive shell startup files.

## Start, inspect, and stop

Run these commands as the user who owns the installation, without `sudo`:

```bash
systemctl --user daemon-reload
systemctl --user enable --now textgen.service
systemctl --user status textgen.service
journalctl --user -u textgen.service -f
```

Open `http://127.0.0.1:7860` in your browser, or the port configured by `--listen-port`. The service does not add `--listen` or enable remote access.

After editing the unit, run `systemctl --user daemon-reload` and `systemctl --user restart textgen.service`. To stop it and disable automatic startup:

```bash
systemctl --user disable --now textgen.service
```

## Start at boot before login

By default, the user service starts with your user manager, normally at login. To start it at boot and keep it running after logout, enable lingering for your account:

```bash
loginctl enable-linger "$USER"
```

This may require administrator authorization, depending on your distribution. To undo it, use `loginctl disable-linger "$USER"`; this affects all services belonging to your user manager, not just textgen.

See the systemd manuals for [user units](https://www.freedesktop.org/software/systemd/man/latest/systemd.unit.html), [service execution](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html), and [lingering](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html).
