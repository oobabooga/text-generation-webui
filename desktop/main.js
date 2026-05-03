const { app, BrowserWindow, screen } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const net = require("net");

const TITLE = "TextGen";
const STARTUP_TIMEOUT_MS = 120000;
const isWin = process.platform === "win32";
const baseDir = app.getAppPath();
const python = path.join(
  baseDir,
  "portable_env",
  isWin ? "python.exe" : path.join("bin", "python3"),
);

// Launcher passes user args after "--" so Chromium's argv parser ignores them.
const argv = process.argv.slice(2);
const dashIdx = argv.indexOf("--");
const userArgs = dashIdx >= 0 ? argv.slice(dashIdx + 1) : argv;

app.setName(TITLE);

// Skip Chromium's hardware video pipeline, which probes VAAPI at startup and
// logs a noisy version-mismatch error on systems with older libva. We don't
// render video content anyway. (--no-sandbox / --no-zygote are passed by the
// launcher script — they must be on the actual argv, not appendSwitch.)
if (process.platform === "linux") {
  app.commandLine.appendSwitch("disable-accelerated-video-decode");
  app.commandLine.appendSwitch("disable-accelerated-video-encode");
}

// Mirrors resolve_user_data_dir in modules/paths.py: --user-data-dir wins,
// else a sibling-level user_data (shared across installs), else in-tree.
function resolveUserDataDir() {
  for (let i = 0; i < userArgs.length; i++) {
    if (userArgs[i] === "--user-data-dir" && i + 1 < userArgs.length) return path.resolve(baseDir, userArgs[i + 1]);
    if (userArgs[i].startsWith("--user-data-dir=")) return path.resolve(baseDir, userArgs[i].slice("--user-data-dir=".length));
  }
  const shared = path.join(baseDir, "..", "..", "user_data");
  return fs.existsSync(shared) ? shared : path.join(baseDir, "..", "user_data");
}
const stateFile = path.join(resolveUserDataDir(), "cache", "window-state.json");

let serverProcess = null;
let mainWindow = null;
let portCheckInterval = null;
let portCheckTimeout = null;

function loadState() {
  try { return JSON.parse(fs.readFileSync(stateFile, "utf8")); } catch { return null; }
}

function saveState() {
  const state = { ...mainWindow.getNormalBounds(), maximized: mainWindow.isMaximized() };
  try {
    fs.mkdirSync(path.dirname(stateFile), { recursive: true });
    fs.writeFileSync(stateFile, JSON.stringify(state));
  } catch {}
}

function checkPort(port) {
  return new Promise((resolve) => {
    const sock = new net.Socket();
    sock.setTimeout(500);
    sock.once("connect", () => { sock.destroy(); resolve(true); });
    sock.once("error", () => resolve(false));
    sock.once("timeout", () => { sock.destroy(); resolve(false); });
    sock.connect(port, "127.0.0.1");
  });
}

function clearTimers() {
  if (portCheckTimeout) { clearTimeout(portCheckTimeout); portCheckTimeout = null; }
  if (portCheckInterval) { clearInterval(portCheckInterval); portCheckInterval = null; }
}

function killServer() {
  const proc = serverProcess;
  if (!proc) return;
  serverProcess = null;
  try {
    if (isWin) {
      spawn("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      process.kill(-proc.pid, "SIGINT");
      setTimeout(() => {
        try { process.kill(-proc.pid, "SIGKILL"); } catch (_) {}
      }, 5000);
    }
  } catch (_) {
    try { proc.kill("SIGINT"); } catch (_) {}
  }
}

function defaultBounds() {
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  return {
    width: Math.min(Math.max(Math.floor(sw * 0.9), 1200), 1600),
    height: Math.min(Math.max(Math.floor(sh * 0.9), 800), 1000),
  };
}

function createWindow(port) {
  const state = loadState();
  const bounds = state && [state.x, state.y, state.width, state.height].every(Number.isFinite)
    ? { x: state.x, y: state.y, width: state.width, height: state.height }
    : defaultBounds();

  mainWindow = new BrowserWindow({
    ...bounds,
    title: TITLE,
    autoHideMenuBar: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });
  if (state && state.maximized) mainWindow.maximize();
  mainWindow.on("page-title-updated", (e) => e.preventDefault());
  mainWindow.webContents.on("will-prevent-unload", (e) => e.preventDefault());
  mainWindow.on("close", saveState);
  mainWindow.on("closed", () => { mainWindow = null; });
  mainWindow.loadURL(`http://127.0.0.1:${port}`);
}

async function waitForPortAndOpen(port) {
  if (await checkPort(port)) {
    createWindow(port);
    return;
  }
  portCheckTimeout = setTimeout(() => {
    clearTimers();
    console.error(`Server failed to become ready within ${STARTUP_TIMEOUT_MS / 1000}s.`);
    app.quit();
  }, STARTUP_TIMEOUT_MS);
  portCheckInterval = setInterval(async () => {
    if (await checkPort(port)) {
      clearTimers();
      createWindow(port);
    }
  }, 500);
}

app.whenReady().then(() => {
  serverProcess = spawn(python, ["server.py", "--portable", "--api", ...userArgs], {
    cwd: baseDir,
    detached: !isWin,
    env: {
      ...process.env,
      PYTHONNOUSERSITE: "1",
      PYTHONPATH: undefined,
      PYTHONHOME: undefined,
      PYTHONUNBUFFERED: "1",
      FORCE_COLOR: "1",
      TERM: "xterm-256color",
    },
  });
  if (!isWin) serverProcess.unref();

  const passthrough = (data) => process.stdout.write(data);
  const onData = (data) => {
    const text = data.toString();
    process.stdout.write(text);
    if (!text.includes("Running on local URL:")) return;
    const match = text.match(/http:\/\/127\.0\.0\.1:(\d+)/);
    if (!match) return;
    serverProcess.stdout.off("data", onData);
    serverProcess.stderr.off("data", onData);
    serverProcess.stdout.on("data", passthrough);
    serverProcess.stderr.on("data", passthrough);
    waitForPortAndOpen(parseInt(match[1], 10));
  };
  serverProcess.stdout.on("data", onData);
  serverProcess.stderr.on("data", onData);

  serverProcess.on("error", (err) => {
    console.error("Failed to spawn server:", err);
    clearTimers();
    app.quit();
  });

  serverProcess.on("close", (code) => {
    console.log(`Server process exited with code ${code}`);
    clearTimers();
    serverProcess = null;
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.close();
    app.quit();
  });
});

app.on("before-quit", killServer);
app.on("window-all-closed", () => app.quit());
process.on("SIGINT", () => { killServer(); process.exit(); });
process.on("SIGTERM", () => { killServer(); process.exit(); });
