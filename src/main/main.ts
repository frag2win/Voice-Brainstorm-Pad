/**
 * Electron main process — Voice Brainstorm Pad Iteration 1
 *
 * Responsibilities:
 *  - Create BrowserWindow with secure defaults
 *  - Expose a narrow IPC API to the renderer via preload
 *  - Delegate all session I/O to SessionManager
 *  - Register clean-shutdown handler
 */

import { app, BrowserWindow, ipcMain, shell } from "electron";
import * as path from "path";
import * as url from "url";
import { SessionManager, SessionMeta } from "./sessions";

// ─── Constants ────────────────────────────────────────────────────────────────

const IS_DEV = process.argv.includes("--dev");
const PRELOAD = path.join(__dirname, "preload.js");
const INDEX_HTML = path.join(__dirname, "../../src/renderer/index.html");

// ─── Global state ─────────────────────────────────────────────────────────────

let win: BrowserWindow | null = null;
let sm: SessionManager | null = null;
let currentSessionSlug: string | null = null;

// ─── Window ───────────────────────────────────────────────────────────────────

function createWindow(): void {
  sm = new SessionManager();

  win = new BrowserWindow({
    width: 1100,
    height: 780,
    minWidth: 600,
    minHeight: 400,
    title: "Voice Brainstorm Pad",
    backgroundColor: "#1a1a2e",
    show: false,
    webPreferences: {
      preload: PRELOAD,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    },
  });

  // Load the renderer HTML directly from disk (no dev-server needed)
  win.loadFile(INDEX_HTML);

  win.once("ready-to-show", () => win?.show());

  win.on("closed", () => {
    win = null;
  });

  // Open external links in the system browser, not in Electron
  win.webContents.setWindowOpenHandler(({ url: href }) => {
    shell.openExternal(href);
    return { action: "deny" };
  });

  if (IS_DEV) {
    win.webContents.openDevTools({ mode: "detach" });
  }
}

// ─── IPC handlers ─────────────────────────────────────────────────────────────

/** sessions:list → SessionMeta[] */
ipcMain.handle("sessions:list", () => {
  return sm?.listSessions() ?? [];
});

/** sessions:create  { title } → SessionMeta */
ipcMain.handle("sessions:create", (_ev, { title }: { title: string }) => {
  const session = sm!.createSession(title?.trim() || "Untitled");
  currentSessionSlug = session.meta.slug;
  return { meta: session.meta, text: session.text };
});

/** sessions:open  { slug } → { meta, text } | null */
ipcMain.handle("sessions:open", (_ev, { slug }: { slug: string }) => {
  const session = sm?.openSession(slug);
  if (!session) return null;
  currentSessionSlug = slug;
  return { meta: session.meta, text: session.text };
});

/** sessions:save  { slug, text } → void */
ipcMain.handle("sessions:save", (_ev, { slug, text }: { slug: string; text: string }) => {
  sm?.saveText(slug, text);
});

/** sessions:rename  { slug, title } → void */
ipcMain.handle("sessions:rename", (_ev, { slug, title }: { slug: string; title: string }) => {
  sm?.renameSession(slug, title);
});

/** sessions:delete  { slug } → void */
ipcMain.handle("sessions:delete", (_ev, { slug }: { slug: string }) => {
  if (slug === currentSessionSlug) currentSessionSlug = null;
  sm?.deleteSession(slug);
});

/** sessions:rootPath → string */
ipcMain.handle("sessions:rootPath", () => sm?.getRoot() ?? "");

// ─── App lifecycle ────────────────────────────────────────────────────────────

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  // Mark clean shutdown for current session
  if (currentSessionSlug && sm) {
    sm.markCleanShutdown(currentSessionSlug);
  }
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  if (currentSessionSlug && sm) {
    sm.markCleanShutdown(currentSessionSlug);
  }
});
