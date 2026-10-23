/**
 * Renderer — Voice Brainstorm Pad Iteration 1
 *
 * Responsibilities:
 *  - Initialise CodeMirror 6 with full text-editor extensions
 *  - Handle session create / open / list / delete
 *  - Debounced autosave (≤ 1 s after last change) via window.vbp
 *  - Live word-count and autosave indicator
 *  - Title editing with debounced rename
 *  - Keyboard shortcuts: Ctrl+N (new), Ctrl+S (force-save), Ctrl+F (find)
 */

import { EditorState, Compartment } from "@codemirror/state";
import { EditorView, keymap, lineNumbers, highlightActiveLineGutter, drawSelection, dropCursor, rectangularSelection, crosshairCursor, highlightActiveLine, placeholder } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, undo, redo, undoDepth } from "@codemirror/commands";
import { search, searchKeymap, openSearchPanel } from "@codemirror/search";
import { markdown } from "@codemirror/lang-markdown";
import { indentOnInput, syntaxHighlighting, defaultHighlightStyle } from "@codemirror/language";

interface SidecarStatus {
  state: string;
  model: string;
  devices: Array<{ id: string; name: string }>;
  port: number;
}

declare global {
  interface Window {
    vbp: {
      sessions: {
        list(): Promise<SessionMeta[]>;
        create(title: string): Promise<SessionPayload>;
        open(slug: string): Promise<SessionPayload | null>;
        save(slug: string, text: string): Promise<void>;
        rename(slug: string, title: string): Promise<void>;
        delete(slug: string): Promise<void>;
        rootPath(): Promise<string>;
      };
      sidecar: {
        getStatus(): Promise<SidecarStatus>;
        restart(): Promise<void>;
        onStateChange(callback: (state: string) => void): () => void;
      };
    };
  }
}

interface SessionMeta {
  slug: string;
  title: string;
  created: string;
  updated: string;
  clean_shutdown: boolean;
}

interface SessionPayload {
  meta: SessionMeta;
  text: string;
}

// ─── App state ────────────────────────────────────────────────────────────────

let editorView: EditorView | null = null;
let currentSlug: string | null = null;
let sessions: SessionMeta[] = [];
let saveTimer: ReturnType<typeof setTimeout> | null = null;
let titleTimer: ReturnType<typeof setTimeout> | null = null;
let isDirty = false;

const AUTOSAVE_DELAY_MS = 800; // save ≤ 1 s after last keystroke

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const elSessionList   = document.getElementById("session-list")!;
const elEditorMount   = document.getElementById("editor-mount")!;
const elEmptyState    = document.getElementById("empty-state")!;
const elSessionTitle  = document.getElementById("session-title") as HTMLInputElement;
const elAutosave      = document.getElementById("autosave-indicator")!;
const elWordCount     = document.getElementById("word-count")!;
const elBtnNew        = document.getElementById("btn-new-session")!;
const elBtnEmptyNew   = document.getElementById("btn-empty-new")!;
const elDialog        = document.getElementById("dialog-new-session") as HTMLDialogElement;
const elFormNew       = document.getElementById("form-new-session") as HTMLFormElement;
const elInputTitle    = document.getElementById("input-new-title") as HTMLInputElement;
const elBtnCancel     = document.getElementById("btn-dialog-cancel")!;
const elRootLabel     = document.getElementById("sessions-root-label")!;
const elSidecarPill   = document.getElementById("sidecar-status-pill")!;
const elSidecarText   = document.getElementById("sidecar-status-text")!;

// ─── CodeMirror setup ─────────────────────────────────────────────────────────

function buildEditorState(text: string): EditorState {
  return EditorState.create({
    doc: text,
    extensions: [
      // History (undo/redo)
      history(),
      // Gutter with line numbers
      lineNumbers(),
      highlightActiveLineGutter(),
      // Cursor and selections
      drawSelection(),
      dropCursor(),
      rectangularSelection(),
      crosshairCursor(),
      // Syntax
      markdown(),
      indentOnInput(),
      syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
      // Active line highlight
      highlightActiveLine(),
      // Placeholder shown when doc is empty
      placeholder("Start brainstorming…"),
      // Search panel
      search({ top: false }),
      // All keybindings
      keymap.of([
        ...defaultKeymap,
        ...historyKeymap,
        ...searchKeymap,
      ]),
      // Line wrapping: essential for long spoken sentences
      EditorView.lineWrapping,
      // Update listener → autosave + word count
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          onDocumentChanged(update.state.doc.toString());
        }
      }),
    ],
  });
}

function mountEditor(text: string): void {
  if (editorView) {
    editorView.destroy();
    editorView = null;
  }

  editorView = new EditorView({
    state: buildEditorState(text),
    parent: elEditorMount,
  });

  editorView.focus();
}

// ─── Document change handler ──────────────────────────────────────────────────

function onDocumentChanged(text: string): void {
  isDirty = true;
  setAutosaveState("saving");
  updateWordCount(text);
  scheduleAutosave(text);
}

function updateWordCount(text: string): void {
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  elWordCount.textContent = words === 1 ? "1 word" : `${words} words`;
}

function scheduleAutosave(text: string): void {
  if (saveTimer !== null) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    if (currentSlug) {
      window.vbp.sessions.save(currentSlug, text).then(() => {
        isDirty = false;
        setAutosaveState("saved");
        // Refresh the session list item's timestamp
        refreshSessionInList(currentSlug!);
        // Fade the "Saved" indicator back out after 2 s
        setTimeout(() => setAutosaveState(""), 2000);
      });
    }
  }, AUTOSAVE_DELAY_MS);
}

function setAutosaveState(state: "saving" | "saved" | ""): void {
  elAutosave.className = state;
  elAutosave.textContent =
    state === "saving" ? "Saving…" :
    state === "saved"  ? "Saved"   : "";
}

// ─── Session list rendering ───────────────────────────────────────────────────

function renderSessionList(): void {
  elSessionList.innerHTML = "";
  if (sessions.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No sessions yet";
    li.style.cssText = "padding:12px 14px;color:var(--text-muted);font-size:12px;";
    elSessionList.appendChild(li);
    return;
  }
  for (const meta of sessions) {
    elSessionList.appendChild(buildSessionItem(meta));
  }
}

function buildSessionItem(meta: SessionMeta): HTMLLIElement {
  const li = document.createElement("li");
  li.className = "session-item" + (meta.slug === currentSlug ? " active" : "");
  li.setAttribute("role", "option");
  li.setAttribute("aria-selected", meta.slug === currentSlug ? "true" : "false");
  li.dataset.slug = meta.slug;

  const title = document.createElement("span");
  title.className = "session-item__title";
  title.textContent = meta.title || "Untitled";

  const metaEl = document.createElement("span");
  metaEl.className = "session-item__meta";
  metaEl.textContent = formatRelative(meta.updated);

  const del = document.createElement("button");
  del.className = "session-item__delete";
  del.title = "Delete session";
  del.setAttribute("aria-label", `Delete session: ${meta.title}`);
  del.textContent = "×";
  del.addEventListener("click", (e) => {
    e.stopPropagation();
    handleDeleteSession(meta.slug);
  });

  li.appendChild(title);
  li.appendChild(metaEl);
  li.appendChild(del);

  li.addEventListener("click", () => handleOpenSession(meta.slug));

  return li;
}

function refreshSessionInList(slug: string): void {
  // Pull updated meta from the server (indirectly: re-fetch list)
  window.vbp.sessions.list().then((list) => {
    sessions = list;
    renderSessionList();
  });
}

function markActiveInList(slug: string | null): void {
  document.querySelectorAll<HTMLLIElement>(".session-item").forEach((li) => {
    const active = li.dataset.slug === slug;
    li.classList.toggle("active", active);
    li.setAttribute("aria-selected", active ? "true" : "false");
  });
}

// ─── Session actions ──────────────────────────────────────────────────────────

async function handleOpenSession(slug: string): Promise<void> {
  if (slug === currentSlug) return;

  // Save any pending changes before switching
  if (currentSlug && isDirty && editorView) {
    await window.vbp.sessions.save(currentSlug, editorView.state.doc.toString());
    isDirty = false;
  }

  const payload = await window.vbp.sessions.open(slug);
  if (!payload) return;

  currentSlug = slug;
  showEditor(payload);
  markActiveInList(slug);
}

async function handleCreateSession(title: string): Promise<void> {
  const payload = await window.vbp.sessions.create(title);
  currentSlug = payload.meta.slug;
  sessions = await window.vbp.sessions.list();
  renderSessionList();
  showEditor(payload);
}

async function handleDeleteSession(slug: string): Promise<void> {
  const meta = sessions.find((s) => s.slug === slug);
  const name = meta?.title || "Untitled";
  if (!confirm(`Delete session "${name}"? This cannot be undone.`)) return;

  await window.vbp.sessions.delete(slug);
  sessions = await window.vbp.sessions.list();

  if (slug === currentSlug) {
    currentSlug = null;
    hideEditor();
  }

  renderSessionList();
}

// ─── Editor visibility ────────────────────────────────────────────────────────

function showEditor(payload: SessionPayload): void {
  elEmptyState.style.display = "none";
  elEditorMount.style.display = "block";
  elSessionTitle.disabled = false;
  elSessionTitle.value = payload.meta.title;

  mountEditor(payload.text);
  updateWordCount(payload.text);
  setAutosaveState("");
}

function hideEditor(): void {
  elEditorMount.style.display = "none";
  elEmptyState.style.display = "";
  elSessionTitle.disabled = true;
  elSessionTitle.value = "";
  elWordCount.textContent = "0 words";
  setAutosaveState("");

  if (editorView) {
    editorView.destroy();
    editorView = null;
  }
}

// ─── Title editing ────────────────────────────────────────────────────────────

elSessionTitle.addEventListener("input", () => {
  if (!currentSlug) return;
  const title = elSessionTitle.value.trim() || "Untitled";
  if (titleTimer !== null) clearTimeout(titleTimer);
  titleTimer = setTimeout(async () => {
    titleTimer = null;
    await window.vbp.sessions.rename(currentSlug!, title);
    // Update list
    const idx = sessions.findIndex((s) => s.slug === currentSlug);
    if (idx >= 0) { sessions[idx].title = title; renderSessionList(); }
  }, 500);
});

elSessionTitle.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    elSessionTitle.blur();
    editorView?.focus();
  }
});

// ─── New session dialog ───────────────────────────────────────────────────────

function openNewDialog(): void {
  elInputTitle.value = "";
  elDialog.showModal();
  elInputTitle.focus();
}

elBtnNew.addEventListener("click", openNewDialog);
elBtnEmptyNew.addEventListener("click", openNewDialog);

elBtnCancel.addEventListener("click", () => elDialog.close("cancel"));

elFormNew.addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = elInputTitle.value.trim() || "Untitled";
  elDialog.close("create");
  await handleCreateSession(title);
});

// Close on backdrop click
elDialog.addEventListener("click", (e) => {
  const rect = elDialog.getBoundingClientRect();
  if (
    e.clientX < rect.left || e.clientX > rect.right ||
    e.clientY < rect.top  || e.clientY > rect.bottom
  ) {
    elDialog.close("cancel");
  }
});

// ─── Global keyboard shortcuts ────────────────────────────────────────────────

document.addEventListener("keydown", async (e) => {
  // Ctrl+N — new session
  if (e.ctrlKey && e.key === "n") {
    e.preventDefault();
    openNewDialog();
    return;
  }

  // Ctrl+S — force-save immediately
  if (e.ctrlKey && e.key === "s") {
    e.preventDefault();
    if (currentSlug && editorView) {
      const text = editorView.state.doc.toString();
      setAutosaveState("saving");
      await window.vbp.sessions.save(currentSlug, text);
      isDirty = false;
      setAutosaveState("saved");
      setTimeout(() => setAutosaveState(""), 2000);
    }
    return;
  }
});

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatRelative(iso: string): string {
  const now = Date.now();
  const t = new Date(iso).getTime();
  const diff = now - t;
  if (diff < 60_000)       return "just now";
  if (diff < 3_600_000)    return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000)   return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function updateSidecarStatus(state: string, model?: string): void {
  if (!elSidecarPill || !elSidecarText) return;
  elSidecarPill.className = `pill pill-${state}`;
  let label = "Connecting…";
  let tooltip = "Speech engine connection status";

  switch (state) {
    case "ready":
    case "idle":
      label = "Sidecar ready";
      tooltip = `Speech sidecar connected and idle (${model || "small.en"})`;
      break;
    case "connecting":
      label = "Connecting…";
      tooltip = "Starting speech sidecar…";
      break;
    case "reconnecting":
      label = "Reconnecting…";
      tooltip = "Speech sidecar restarted, reconnecting…";
      break;
    case "disconnected":
      label = "Sidecar offline";
      tooltip = "Speech sidecar is offline (typing & editor remain fully functional)";
      break;
    case "error":
      label = "Sidecar error";
      tooltip = "Speech sidecar error";
      break;
    default:
      label = state;
      tooltip = `Sidecar state: ${state}`;
  }

  elSidecarText.textContent = label;
  elSidecarPill.title = tooltip;
}

// ─── Init ─────────────────────────────────────────────────────────────────────

async function init(): Promise<void> {
  // Populate sessions root path in footer
  const rootPath = await window.vbp.sessions.rootPath();
  elRootLabel.textContent = rootPath;
  elRootLabel.title = rootPath;

  // Initialize sidecar status tracking
  try {
    const checkStatus = async () => {
      const status = await window.vbp.sidecar.getStatus();
      updateSidecarStatus(status.state, status.model);
      return status.state;
    };

    const initialState = await checkStatus();
    if (initialState === "connecting" || initialState === "disconnected") {
      const pollTimer = setInterval(async () => {
        const st = await checkStatus();
        if (st !== "connecting" && st !== "disconnected") {
          clearInterval(pollTimer);
        }
      }, 500);
      setTimeout(() => clearInterval(pollTimer), 10000);
    }

    window.vbp.sidecar.onStateChange((state) => {
      updateSidecarStatus(state);
    });
  } catch (err) {
    console.warn("Could not query sidecar status:", err);
  }

  // Load session list
  sessions = await window.vbp.sessions.list();
  renderSessionList();

  // If there are sessions, open the most recent one automatically
  if (sessions.length > 0) {
    await handleOpenSession(sessions[0].slug);
  }
}

init().catch(console.error);

