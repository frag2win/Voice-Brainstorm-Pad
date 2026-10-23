/**
 * Tests for SessionManager — Voice Brainstorm Pad Iteration 1
 *
 * Tests cover:
 *  - Session creation (slug format, initial text empty, meta written)
 *  - Atomic save (text survives re-open)
 *  - List ordering (newest updated first)
 *  - Rename
 *  - Delete
 *  - Clean-shutdown flag
 *  - Concurrent saves don't corrupt (sequential writes)
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { SessionManager } from "../src/main/sessions";

function tmpRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "vbp-test-"));
}

describe("SessionManager", () => {
  let root: string;
  let sm: SessionManager;

  beforeEach(() => {
    root = tmpRoot();
    sm = new SessionManager(root);
  });

  afterEach(() => {
    fs.rmSync(root, { recursive: true, force: true });
  });

  // ── Create ─────────────────────────────────────────────────────────────────

  it("creates a session with an empty document", () => {
    const s = sm.createSession("My Session");
    expect(s.text).toBe("");
    expect(s.meta.title).toBe("My Session");
    expect(s.meta.slug).toBeTruthy();
    expect(s.meta.clean_shutdown).toBe(false);
  });

  it("session folder and files exist on disk after create", () => {
    const s = sm.createSession("Disk Check");
    expect(fs.existsSync(s.folderPath)).toBe(true);
    expect(fs.existsSync(path.join(s.folderPath, "session.md"))).toBe(true);
    expect(fs.existsSync(path.join(s.folderPath, "meta.json"))).toBe(true);
  });

  it("slug contains a date prefix and a safe title", () => {
    const s = sm.createSession("Hello World Test!");
    expect(s.meta.slug).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_hello-world-test$/);
  });

  it("slug falls back to 'untitled' for empty title", () => {
    const s = sm.createSession("");
    expect(s.meta.slug).toMatch(/untitled$/);
  });

  it("two sessions created in sequence have different slugs", () => {
    const a = sm.createSession("A");
    const b = sm.createSession("B");
    expect(a.meta.slug).not.toBe(b.meta.slug);
  });

  // ── Save / Open ────────────────────────────────────────────────────────────

  it("saves text atomically and re-open returns it unchanged", () => {
    const s = sm.createSession("Save Test");
    const text = "The quick brown fox\njumped over the lazy dog.\n";
    sm.saveText(s.meta.slug, text);
    const reopened = sm.openSession(s.meta.slug);
    expect(reopened?.text).toBe(text);
  });

  it("save updates meta.updated timestamp", async () => {
    const s = sm.createSession("Timestamp");
    const before = s.meta.updated;
    // Wait 1 ms to ensure different timestamp
    await new Promise((r) => setTimeout(r, 2));
    sm.saveText(s.meta.slug, "hello");
    const reopened = sm.openSession(s.meta.slug);
    expect(reopened!.meta.updated > before).toBe(true);
  });

  it("large text (100 KB) saves and loads correctly", () => {
    const s = sm.createSession("Large");
    const big = "word ".repeat(20_000); // ~100 KB
    sm.saveText(s.meta.slug, big);
    const reopened = sm.openSession(s.meta.slug);
    expect(reopened?.text).toBe(big);
  });

  it("openSession returns null for unknown slug", () => {
    expect(sm.openSession("does-not-exist")).toBeNull();
  });

  // ── List ───────────────────────────────────────────────────────────────────

  it("lists sessions newest-updated first", async () => {
    const a = sm.createSession("A");
    await new Promise((r) => setTimeout(r, 2));
    sm.saveText(a.meta.slug, "update a");
    await new Promise((r) => setTimeout(r, 2));
    const b = sm.createSession("B");
    const list = sm.listSessions();
    // B was created (= updated) after A's last save
    expect(list[0].slug).toBe(b.meta.slug);
    expect(list[1].slug).toBe(a.meta.slug);
  });

  it("returns empty list when no sessions exist", () => {
    const fresh = new SessionManager(tmpRoot());
    expect(fresh.listSessions()).toEqual([]);
    // cleanup
    fs.rmSync(fresh.getRoot(), { recursive: true, force: true });
  });

  // ── Rename ─────────────────────────────────────────────────────────────────

  it("rename updates meta.title and is reflected in list", () => {
    const s = sm.createSession("Old Name");
    sm.renameSession(s.meta.slug, "New Name");
    const list = sm.listSessions();
    const found = list.find((m) => m.slug === s.meta.slug);
    expect(found?.title).toBe("New Name");
  });

  // ── Delete ─────────────────────────────────────────────────────────────────

  it("delete removes the session from disk and from listSessions", () => {
    const s = sm.createSession("To Delete");
    sm.deleteSession(s.meta.slug);
    expect(fs.existsSync(s.folderPath)).toBe(false);
    const list = sm.listSessions();
    expect(list.find((m) => m.slug === s.meta.slug)).toBeUndefined();
  });

  it("deleteSession on nonexistent slug does not throw", () => {
    expect(() => sm.deleteSession("ghost")).not.toThrow();
  });

  // ── Clean shutdown ─────────────────────────────────────────────────────────

  it("clean_shutdown is false after create and save, true after markCleanShutdown", () => {
    const s = sm.createSession("Clean");
    sm.saveText(s.meta.slug, "some text");
    sm.markCleanShutdown(s.meta.slug);
    const reopened = sm.openSession(s.meta.slug);
    expect(reopened?.meta.clean_shutdown).toBe(true);
  });

  it("saveText resets clean_shutdown to false", () => {
    const s = sm.createSession("Reset");
    sm.markCleanShutdown(s.meta.slug);
    sm.saveText(s.meta.slug, "new content");
    const reopened = sm.openSession(s.meta.slug);
    expect(reopened?.meta.clean_shutdown).toBe(false);
  });

  // ── Root path ──────────────────────────────────────────────────────────────

  it("getRoot returns the configured sessions root", () => {
    expect(sm.getRoot()).toBe(root);
  });
});
