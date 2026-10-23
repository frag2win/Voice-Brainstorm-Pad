/**
 * Session management for Voice Brainstorm Pad — Iteration 1
 *
 * Owns all disk I/O for sessions. Each session lives in:
 *   <rootDir>/<slug>/
 *     session.md     — the document text (atomically written)
 *     meta.json      — metadata (title, created, updated, clean_shutdown)
 *
 * Nothing in this module knows about Electron or the UI.
 */

import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as crypto from "crypto";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SessionMeta {
  /** Slug is the directory name under sessionsRoot */
  slug: string;
  title: string;
  created: string; // ISO-8601
  updated: string; // ISO-8601
  /** True only if the app exited cleanly (written on shutdown) */
  clean_shutdown: boolean;
}

export interface Session {
  meta: SessionMeta;
  text: string;
  /** Absolute path to the session folder */
  folderPath: string;
}

// ─── Paths ────────────────────────────────────────────────────────────────────

/**
 * Default sessions root: ~/Documents/VoiceBrainstormPad/sessions
 * Can be overridden by the app at startup.
 */
export function defaultSessionsRoot(): string {
  return path.join(os.homedir(), "Documents", "VoiceBrainstormPad", "sessions");
}

// ─── Session Manager ──────────────────────────────────────────────────────────

export class SessionManager {
  private sessionsRoot: string;

  constructor(sessionsRoot?: string) {
    this.sessionsRoot = sessionsRoot ?? defaultSessionsRoot();
    fs.mkdirSync(this.sessionsRoot, { recursive: true });
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private folderFor(slug: string): string {
    return path.join(this.sessionsRoot, slug);
  }

  private textPath(slug: string): string {
    return path.join(this.folderFor(slug), "session.md");
  }

  private metaPath(slug: string): string {
    return path.join(this.folderFor(slug), "meta.json");
  }

  /**
   * Atomic write: write to a temp file beside the target, then rename.
   * On Windows, fs.renameSync is atomic within the same volume.
   */
  private atomicWrite(filePath: string, content: string): void {
    const dir = path.dirname(filePath);
    const tmp = path.join(dir, `.tmp_${crypto.randomBytes(6).toString("hex")}`);
    fs.writeFileSync(tmp, content, { encoding: "utf8", flag: "w" });
    fs.renameSync(tmp, filePath);
  }

  private readMeta(slug: string): SessionMeta | null {
    const mp = this.metaPath(slug);
    if (!fs.existsSync(mp)) return null;
    try {
      const raw = fs.readFileSync(mp, "utf8");
      return JSON.parse(raw) as SessionMeta;
    } catch {
      return null;
    }
  }

  private writeMeta(meta: SessionMeta): void {
    const dir = this.folderFor(meta.slug);
    fs.mkdirSync(dir, { recursive: true });
    this.atomicWrite(this.metaPath(meta.slug), JSON.stringify(meta, null, 2));
  }

  private generateSlug(title: string): string {
    const now = new Date();
    const date = now.toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const safe = title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "untitled";
    return `${date}_${safe}`;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Create a brand-new session with an empty document.
   * Returns the full session object.
   */
  createSession(title: string = "Untitled"): Session {
    const now = new Date().toISOString();
    const slug = this.generateSlug(title);
    const meta: SessionMeta = {
      slug,
      title,
      created: now,
      updated: now,
      clean_shutdown: false,
    };
    const dir = this.folderFor(slug);
    fs.mkdirSync(dir, { recursive: true });
    this.writeMeta(meta);
    this.atomicWrite(this.textPath(slug), "");
    return { meta, text: "", folderPath: dir };
  }

  /**
   * Open an existing session by slug.
   * Returns null if the session does not exist.
   */
  openSession(slug: string): Session | null {
    const meta = this.readMeta(slug);
    if (!meta) return null;
    const tp = this.textPath(slug);
    const text = fs.existsSync(tp) ? fs.readFileSync(tp, "utf8") : "";
    return { meta, text, folderPath: this.folderFor(slug) };
  }

  /**
   * Save the current document text atomically.
   * Also updates meta.updated and clears clean_shutdown.
   */
  saveText(slug: string, text: string): void {
    this.atomicWrite(this.textPath(slug), text);
    const meta = this.readMeta(slug);
    if (meta) {
      meta.updated = new Date().toISOString();
      meta.clean_shutdown = false;
      this.writeMeta(meta);
    }
  }

  /**
   * Update just the session title.
   */
  renameSession(slug: string, newTitle: string): void {
    const meta = this.readMeta(slug);
    if (!meta) return;
    meta.title = newTitle;
    meta.updated = new Date().toISOString();
    this.writeMeta(meta);
  }

  /**
   * Mark a clean shutdown so recovery knows to skip this session.
   */
  markCleanShutdown(slug: string): void {
    const meta = this.readMeta(slug);
    if (!meta) return;
    meta.clean_shutdown = true;
    this.writeMeta(meta);
  }

  /**
   * List all sessions, newest first, by meta.updated.
   */
  listSessions(): SessionMeta[] {
    if (!fs.existsSync(this.sessionsRoot)) return [];
    const dirs = fs
      .readdirSync(this.sessionsRoot, { withFileTypes: true })
      .filter((d) => d.isDirectory())
      .map((d) => d.name);

    const metas: SessionMeta[] = [];
    for (const slug of dirs) {
      const m = this.readMeta(slug);
      if (m) metas.push(m);
    }
    return metas.sort(
      (a, b) => new Date(b.updated).getTime() - new Date(a.updated).getTime()
    );
  }

  /**
   * Permanently delete a session and all its files.
   */
  deleteSession(slug: string): void {
    const dir = this.folderFor(slug);
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }

  /** Returns the sessions root so the UI can display it. */
  getRoot(): string {
    return this.sessionsRoot;
  }
}
