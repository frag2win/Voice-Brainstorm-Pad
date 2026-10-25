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
import {
  dictationAnchorField,
  dictationAnchorDecoration,
  getDictationAnchor,
  moveAnchorToCaret,
  jumpToAnchor,
  dictateTextTransaction,
} from "./anchor";
import { formatDictationText, formatChunkFailedNotice } from "./spacing";

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
        startListening(sessionFolder: string, deviceId?: string): Promise<void>;
        stopListening(): Promise<void>;
        setDevice(deviceId: string): Promise<void>;
        onStateChange(callback: (state: string) => void): () => void;
        onLevel(callback: (rms: number) => void): () => void;
        onChunk(callback: (chunk: any) => void): () => void;
        onTranscript(callback: (data: any) => void): () => void;
        onChunkFailed(callback: (data: any) => void): () => void;
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
  folderPath: string;
}

// ─── App state ────────────────────────────────────────────────────────────────

let editorView: EditorView | null = null;
let currentSlug: string | null = null;
let currentSessionFolder: string | null = null;
let currentSidecarState: string = "idle";
let sessions: SessionMeta[] = [];
let saveTimer: ReturnType<typeof setTimeout> | null = null;
let titleTimer: ReturnType<typeof setTimeout> | null = null;
let isDirty = false;
let lastChunkEnd: number | null = null;

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
const elBtnDictate    = document.getElementById("btn-dictate") as HTMLButtonElement;
const elDictateLabel  = document.getElementById("dictate-label")!;
const elDictateIcon   = document.getElementById("dictate-icon")!;
const elLevelBar      = document.getElementById("level-meter-bar")!;
const elMicSelect     = document.getElementById("mic-select") as HTMLSelectElement;
const elBtnMoveAnchor = document.getElementById("btn-move-anchor") as HTMLButtonElement | null;
const elBtnJumpAnchor = document.getElementById("btn-jump-anchor") as HTMLButtonElement | null;

// ─── CodeMirror setup ─────────────────────────────────────────────────────────

function buildEditorState(text: string): EditorState {
  return EditorState.create({
    doc: text,
    extensions: [
      // History (undo/redo)
      history(),
      // Dictation Anchor (Iteration 5)
      dictationAnchorField,
      dictationAnchorDecoration,
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
        {
          key: "Mod-Alt-a",
          run: (v) => {
            moveAnchorToCaret(v);
            return true;
          },
        },
        {
          key: "Mod-Alt-j",
          run: (v) => {
            jumpToAnchor(v);
            return true;
          },
        },
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
  lastChunkEnd = null;
  currentSessionFolder = payload.folderPath;
  elEmptyState.style.display = "none";
  elEditorMount.style.display = "block";
  elSessionTitle.disabled = false;
  elSessionTitle.value = payload.meta.title;

  const isAvailable =
    currentSidecarState !== "disconnected" &&
    currentSidecarState !== "error" &&
    currentSidecarState !== "connecting" &&
    currentSidecarState !== "reconnecting";
  elBtnDictate.disabled = !isAvailable;
  if (elBtnMoveAnchor) elBtnMoveAnchor.disabled = false;
  if (elBtnJumpAnchor) elBtnJumpAnchor.disabled = false;

  mountEditor(payload.text);
  updateWordCount(payload.text);
  setAutosaveState("");
}

function hideEditor(): void {
  lastChunkEnd = null;
  if (currentSidecarState === "listening") {
    window.vbp.sidecar.stopListening().catch(() => {});
  }
  currentSessionFolder = null;
  elEditorMount.style.display = "none";
  elEmptyState.style.display = "";
  elSessionTitle.disabled = true;
  elSessionTitle.value = "";
  elWordCount.textContent = "0 words";
  elBtnDictate.disabled = true;
  if (elBtnMoveAnchor) elBtnMoveAnchor.disabled = true;
  if (elBtnJumpAnchor) elBtnJumpAnchor.disabled = true;
  elBtnDictate.classList.remove("listening");
  elDictateLabel.textContent = "Dictate";
  elDictateIcon.textContent = "🎙️";
  elLevelBar.style.width = "0%";
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
  currentSidecarState = state;
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
    case "listening":
      label = "Listening…";
      tooltip = "Speech sidecar is recording and chunking audio";
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

  if (state === "listening") {
    elBtnDictate.classList.add("listening");
    elDictateLabel.textContent = "Stop";
    elDictateIcon.textContent = "⏹";
  } else {
    elBtnDictate.classList.remove("listening");
    elDictateLabel.textContent = "Dictate";
    elDictateIcon.textContent = "🎙️";
    elLevelBar.style.width = "0%";
  }

  const isAvailable =
    currentSessionFolder !== null &&
    state !== "disconnected" &&
    state !== "error" &&
    state !== "connecting" &&
    state !== "reconnecting";

  elBtnDictate.disabled = !isAvailable;
}

function populateDevices(devices: Array<{ id: string; name: string }>): void {
  if (!devices || devices.length === 0) return;
  const currentVal = elMicSelect.value;
  elMicSelect.innerHTML = '<option value="">Default Microphone</option>';
  for (const dev of devices) {
    const opt = document.createElement("option");
    opt.value = dev.id;
    opt.textContent = dev.name;
    if (dev.id === currentVal) opt.selected = true;
    elMicSelect.appendChild(opt);
  }
}

export function showChunkFailedNotice(seq: number): void {
  let container = document.getElementById("chunk-failed-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "chunk-failed-container";
    container.setAttribute("aria-live", "polite");
    document.body.appendChild(container);
  }
  const notice = document.createElement("div");
  notice.className = "chunk-failed-notice";
  notice.setAttribute("role", "status");
  notice.textContent = formatChunkFailedNotice(seq);
  container.appendChild(notice);

  setTimeout(() => {
    notice.style.opacity = "0";
    notice.style.transform = "translateY(10px)";
    setTimeout(() => {
      notice.remove();
      if (container && container.children.length === 0) {
        container.remove();
      }
    }, 300);
  }, 5000);
}

// ─── Init ─────────────────────────────────────────────────────────────────────

async function init(): Promise<void> {
  // Populate sessions root path in footer
  const rootPath = await window.vbp.sessions.rootPath();
  elRootLabel.textContent = rootPath;
  elRootLabel.title = rootPath;

  // Wire Dictation Toggle Button
  elBtnDictate.addEventListener("click", async () => {
    if (currentSidecarState === "listening") {
      await window.vbp.sidecar.stopListening();
    } else if (currentSessionFolder) {
      lastChunkEnd = null;
      const deviceId = elMicSelect.value || undefined;
      await window.vbp.sidecar.startListening(currentSessionFolder, deviceId);
    }
  });

  // Wire Dictation Anchor Controls (Iteration 5 - FR-6)
  elBtnMoveAnchor?.addEventListener("click", () => {
    if (editorView) {
      moveAnchorToCaret(editorView);
      editorView.focus();
    }
  });

  elBtnJumpAnchor?.addEventListener("click", () => {
    if (editorView) {
      jumpToAnchor(editorView);
    }
  });

  // Wire Microphone Selector
  elMicSelect.addEventListener("change", async () => {
    await window.vbp.sidecar.setDevice(elMicSelect.value);
  });

  // Wire Live Audio Level Meter
  window.vbp.sidecar.onLevel((rms: number) => {
    if (currentSidecarState === "listening") {
      const pct = Math.min(100, Math.max(0, Math.round(Math.sqrt(rms) * 180)));
      elLevelBar.style.width = `${pct}%`;
    }
  });

  // Wire Chunk Events
  window.vbp.sidecar.onChunk((chunk: any) => {
    console.log(`[Chunker] Emitted chunk #${chunk.seq} (${chunk.duration_s}s) -> ${chunk.audio_file}`);
  });

  // Wire Transcript Events (Iteration 5 - Streaming Editor Insertion at Dictation Anchor)
  window.vbp.sidecar.onTranscript((data: any) => {
    console.log(`[Transcriber] #${data.seq} (${data.rtf}x RTF, ${data.words}w): "${data.text}"`);
    if (!editorView || !data.text || !data.text.trim()) return;

    let pauseDurationS = 0;
    if (lastChunkEnd !== null && typeof data.t_start === "number") {
      pauseDurationS = Math.max(0, data.t_start - lastChunkEnd);
    }
    lastChunkEnd = typeof data.t_end === "number"
      ? data.t_end
      : (typeof data.t_start === "number" ? data.t_start : 0) + (data.duration_s || 0);

    const docText = editorView.state.doc.toString();
    const anchor = getDictationAnchor(editorView.state);
    const formatted = formatDictationText(docText, anchor, data.text, pauseDurationS);

    if (formatted.insertText) {
      const trSpec = dictateTextTransaction(editorView.state, formatted.insertText);
      trSpec.scrollIntoView = true;
      editorView.dispatch(trSpec);
    }
  });

  // Wire Chunk Failed Events (Iteration 4.2 Item 9)
  window.vbp.sidecar.onChunkFailed((data: any) => {
    console.warn(`[Transcriber] chunk #${data?.seq} failed: ${data?.error}`);
    if (typeof data?.seq === "number") {
      showChunkFailedNotice(data.seq);
    }
  });

  // Initialize sidecar status tracking
  try {
    const checkStatus = async () => {
      const status = await window.vbp.sidecar.getStatus();
      updateSidecarStatus(status.state, status.model);
      if (status.devices && status.devices.length > 0) {
        populateDevices(status.devices);
      }
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
      if (state === "listening") {
        lastChunkEnd = null;
      }
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

