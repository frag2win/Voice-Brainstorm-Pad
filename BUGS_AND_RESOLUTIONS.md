# Voice Brainstorm Pad — Bug Registry and Resolution Log

This document serves as the master record of all resolved defects, architectural issues, and runtime bugs encountered during development, along with their root causes, fixes, and prevention guidelines.

---

## 1. Resolved Bugs (Master Registry)

### BUG-001: Double Dictation Anchor Icon (Static + Animated)
- **Status**: Resolved
- **Area**: Renderer / Editor Anchor (`anchor.ts`, `style.css`)
- **Symptoms**: In the editor, two anchor icons appeared right next to each other at the dictation insertion point. One was animated with a pulse, while the other was completely static.
- **Root Cause**: `AnchorMarkerWidget.toDOM()` in `anchor.ts` set `span.textContent = "⚓"`, creating a text node glyph. Simultaneously, `style.css` defined `.cm-dictation-anchor::after { content: "⚓"; animation: anchor-pulse ... }`. Both the element body and the CSS pseudo-element were active at the same time.
- **Fix**: Removed `span.textContent = "⚓"` from `AnchorMarkerWidget.toDOM()` in `src/renderer/anchor.ts`. Retained the animated `.cm-dictation-anchor::after` pseudo-element in `style.css` with adjusted inline width (`1.1em`) so it animates cleanly without doubling.
- **Prevention**: When creating visual CodeMirror widgets styled via CSS `::before` or `::after`, do not assign raw textContent to the host element.

---

### BUG-002: Excess Document Margin and Wasted Horizontal Padding
- **Status**: Resolved
- **Area**: Renderer Layout (`style.css`)
- **Symptoms**: The text writing canvas was constricted into a narrow centered box (max 780px) with large black margins on both sides, leaving significant horizontal screen space unused.
- **Root Cause**: `.document-wrapper` was styled with `max-width: 780px; margin: 0 auto;`, and `.editor-scroll-area` had large padding (`40px 24px 160px`), artificially boxing the text.
- **Fix**: Changed `.document-wrapper` to `width: 100%; max-width: none; margin: 0;`, and tightened `.editor-scroll-area` padding to `28px 36px 160px`. Text now seamlessly spans the full width of the editor canvas.
- **Prevention**: Avoid fixed container max-widths in note-taking and editor canvas wrappers unless specifically requested.

---

### BUG-003: New Section Dialog Input Not Clickable / Uneditable
- **Status**: Resolved
- **Area**: Renderer Dialog (`renderer.ts`, `style.css`)
- **Symptoms**: Clicking "New Section" opened a modal, but typing or clicking the section name input failed or caused the dialog to immediately close.
- **Root Cause**:
  1. `body` had `user-select: none;`, which Chromium inherited into `<input>`, disabling cursor placement and text selection.
  2. The backdrop dismissal handler measured `rect = elDialog.getBoundingClientRect()`. Clicking inside the form or input triggered coordinate boundary mismatches, closing the modal immediately.
- **Fix**:
  1. Added `user-select: text !important; -webkit-user-select: text !important;` to all inputs, textareas, and `.cm-content`.
  2. Replaced the bounding rect click check with `if (e.target === elDialog) { elDialog.close("cancel"); }`, and added `e.stopPropagation()` to `form.addEventListener("click")`.
  3. Added auto-focus and text selection timeouts when opening the modal.
- **Prevention**: Always use `e.target === dialog` to detect backdrop clicks on HTML `<dialog>`. Never let global `user-select: none` cascade onto form inputs.

---

### BUG-004: Collapsed Sidebar Rail Rendering "No sessions yet"
- **Status**: Resolved
- **Area**: Renderer Sidebar Rail (`style.css`)
- **Symptoms**: When the sidebar rail was collapsed to 72px, the session list header and empty-state text "No sessions yet" remained visible, awkwardly wrapping into 3 lines ("No \n session \n yet").
- **Root Cause**: `.sidebar-rail.collapsed` did not hide `.sidebar-list-container`.
- **Fix**: Added `.sidebar-rail.collapsed .sidebar-list-container { display: none !important; }`. The collapsed rail now displays only the icon buttons (Menu toggle, New Section pen, Search icon).
- **Prevention**: Whenever a collapsible rail is implemented, all text containers, list items, and headers inside must be strictly hidden when collapsed.

---

### BUG-005: Local System Path Displayed in Sidebar Footer
- **Status**: Resolved
- **Area**: Sidebar UI (`index.html`, `renderer.ts`)
- **Symptoms**: An unnecessary absolute path (`C:\Users\...\Documents\VoiceBrainstormPad\sessions`) was displayed at the bottom of the sidebar.
- **Root Cause**: Leftover template element `#sessions-root-label` in sidebar footer.
- **Fix**: Removed `#sidebar-footer` and `#sessions-root-label` from `src/renderer/index.html` and made `elRootLabel` safely optional in `renderer.ts`.
- **Prevention**: System diagnostic badges should be confined to dev logs or detached settings screens rather than main navigation sidebars.

---

### BUG-006: Electron Window Not Displaying When Launched From Background Shell
- **Status**: Resolved
- **Area**: Electron Main Process (`main.ts`)
- **Symptoms**: Running `npm start` from automated background subprocesses on Windows did not bring up the application window on the interactive desktop.
- **Root Cause**:
  1. `BrowserWindow` was created with `show: false`, waiting indefinitely on `ready-to-show`.
  2. Windows OS session isolation prevents background service child processes from rendering on the interactive desktop window station (`WinSta0\Default`).
- **Fix**:
  1. Set `show: true` directly in `BrowserWindow` creation and added explicit `win.focus()`.
  2. Informed the user to run `npm start` directly within the interactive terminal in VS Code / Windows desktop.
- **Prevention**: Desktop GUI frameworks (Electron, Qt) must be executed from an interactive desktop terminal rather than a headless background agent runner.

---

### BUG-007: MKL Threading OpenMP Memory Leak in Python Sidecar
- **Status**: Resolved
- **Area**: Sidecar (`server.py`, `transcriber.py`)
- **Symptoms**: Memory usage climbed continuously during prolonged dictation sessions.
- **Root Cause**: OpenMP / MKL thread pools were being allocated per chunk inference without thread pool reuse or environment capping.
- **Fix**: Pinned environment variables `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and capped CTranslate2 thread pool concurrency.
- **Prevention**: Set thread limits and disable OpenMP dynamic allocation before importing heavy numerical libraries like PyTorch or CTranslate2.

---

### BUG-008: Catch-up Drain Blocking WebSocket Message Loop
- **Status**: Resolved
- **Area**: Python Sidecar Server (`server.py`)
- **Symptoms**: Stopping dictation with pending queued chunks blocked the WebSocket event loop, causing pings to timeout and supervisor to disconnect.
- **Root Cause**: The chunk drain loop ran synchronously inside the `stop` command handler.
- **Fix**: Converted drain loop to an asynchronous background task `asyncio.create_task(self._drain_then_idle())`, immediately returning control to the WebSocket loop so pings/pongs continue while draining.
- **Prevention**: Never run long loops or blocking drains inside WebSocket event handlers; dispatch background tasks and return to the message loop.

---

## 2. Audio & Transcription Analysis

### Diagnosis: Discrepancy Between Spoken Speech and Transcribed Text
- **Reported Issue**: Spoken speech did not match what appeared in the editor ("What I'm speaking and what it is typing is completely two different things").
- **Audio Measurement Analysis**:
  - Sample session chunk analysis (`C:\Users\sunanda.AMFIIND\Documents\VoiceBrainstormPad\sessions\2026-10-25T17-39-51_test\chunks\chunk_0001.wav` through `chunk_0006.wav`):
    - `chunk_0001.wav`: Peak = 821, RMS = 137.1, **dBFS = -47.6**
    - `chunk_0002.wav`: Peak = 1149, RMS = 151.3, **dBFS = -46.7**
    - `chunk_0003.wav`: Peak = 1794, RMS = 147.2, **dBFS = -47.0**
    - `chunk_0004.wav`: Peak = 1272, RMS = 178.3, **dBFS = -45.3**
    - `chunk_0005.wav`: Peak = 2087, RMS = 201.8, **dBFS = -44.2**
    - `chunk_0006.wav`: Peak = 1607, RMS = 237.3, **dBFS = -42.8**
- **Root Cause**:
  1. **Severely Attenuated Input Signal**: Full-scale 16-bit audio is 32768. The maximum peak recorded was only 821 to 2087 (2.5% to 6.3% of maximum input level). At -47 dBFS, audio is close to the noise floor.
  2. **Whisper Decoder Compensation**: When fed low-amplitude, muffled speech from a laptop microphone array with hardware noise suppression, Whisper boosts gain and decodes phonetically ambiguous syllables into closest English dictionary phrases (e.g., misinterpreting muffled speech as "c for any kind of critical property...").
  3. **Acoustic Mismatch on `small.en`**: `small.en` is an English-only acoustic model trained predominantly on standard near-field speech. When speech is fast and distant from the microphone, word error rates increase significantly.
- **Recommendations & Fixes**:
  1. **Windows Microphone Input Volume**: In Windows Settings > Sound > Input > Properties for "Microphone Array", increase the Input volume slider to 90–100% and ensure Microphone Boost is enabled.
  2. **Microphone Distance**: Position within 30–50 cm of the microphone array when dictating.
  3. **Software Gain Normalization**: Pre-normalize chunk audio levels before running inference if raw input RMS falls below -30 dBFS.

---

## 3. Dictation Anchor Controls Reference

### "Dictate Here" (`#btn-move-anchor`, Shortcut: `Ctrl+Alt+A`)
- **What it does**: Moves the dictation insertion anchor to your current keyboard caret position.
- **When to use it**: If you have written several paragraphs and want newly dictated thoughts inserted into an earlier section or in the middle of a sentence (rather than at the end of the document), click your cursor where you want the text and press "Dictate Here". All subsequent speech will insert right at that marker.

### "Jump to Anchor" (`#btn-jump-anchor`, Shortcut: `Ctrl+Alt+J`)
- **What it does**: Instantly scrolls the editor and places your keyboard text cursor directly at the current dictation anchor.
- **When to use it**: If you are reviewing a long document while dictating and lose track of where the speech is being inserted, clicking "Jump to Anchor" brings the anchor back into view and positions your caret next to it for manual editing.
