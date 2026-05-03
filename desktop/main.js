const { app, BrowserWindow, screen } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
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

let serverProcess = null;
let mainWindow = null;
let portCheckInterval = null;
let portCheckTimeout = null;

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

function createWindow(port) {
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const width = Math.min(Math.max(Math.floor(sw * 0.9), 1200), 1600);
  const height = Math.min(Math.max(Math.floor(sh * 0.9), 800), 1000);

  mainWindow = new BrowserWindow({
    width,
    height,
    title: TITLE,
    autoHideMenuBar: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  });
  mainWindow.on("page-title-updated", (e) => e.preventDefault());
  mainWindow.webContents.on("will-prevent-unload", (e) => e.preventDefault());
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
