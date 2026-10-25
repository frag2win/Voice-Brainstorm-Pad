/**
 * Dictation Anchor for CodeMirror 6 — Voice Brainstorm Pad Iteration 5
 *
 * Implements TRD Section 2.4:
 *  - StateField holding anchor position
 *  - On every transaction, position is mapped with forward association (assoc = 1)
 *  - If typing exactly at anchor, anchor shifts after typed text
 *  - If text around anchor is deleted, anchor collapses to deletion point
 *  - Dictated insertions are tagged with 'input.dictation' userEvent
 *  - Visible anchor marker widget
 *  - Commands: move anchor to caret, jump to anchor
 */

import {
  EditorState,
  StateField,
  StateEffect,
  Transaction,
  TransactionSpec,
  EditorSelection,
} from "@codemirror/state";
import { EditorView, WidgetType, Decoration, DecorationSet } from "@codemirror/view";

// ─── State Effect ─────────────────────────────────────────────────────────────

export const setAnchorEffect = StateEffect.define<number>();

// ─── Anchor StateField ────────────────────────────────────────────────────────

export const dictationAnchorField = StateField.define<number>({
  create(state: EditorState): number {
    return state.doc.length;
  },
  update(currentPos: number, tr: Transaction): number {
    // 1. Check for explicit effect to set anchor
    for (const effect of tr.effects) {
      if (effect.is(setAnchorEffect)) {
        return Math.max(0, Math.min(effect.value, tr.newDoc.length));
      }
    }

    // 2. Map position through document changes
    if (tr.docChanged) {
      // assoc = 1 (forward association): if text is inserted at currentPos,
      // currentPos maps to after the inserted text.
      // If deleted across currentPos, collapses to the start of the deletion.
      const mapped = tr.changes.mapPos(currentPos, 1);
      return Math.max(0, Math.min(mapped, tr.newDoc.length));
    }

    return currentPos;
  },
});

// ─── Helpers ──────────────────────────────────────────────────────────────────

export function getDictationAnchor(state: EditorState): number {
  const pos = state.field(dictationAnchorField, false);
  return typeof pos === "number" ? pos : state.doc.length;
}

export function setDictationAnchor(view: EditorView, pos: number): void {
  view.dispatch({
    effects: [setAnchorEffect.of(pos)],
  });
}

export function moveAnchorToCaret(view: EditorView): void {
  const head = view.state.selection.main.head;
  setDictationAnchor(view, head);
}

export function jumpToAnchor(view: EditorView): void {
  const anchor = getDictationAnchor(view.state);
  view.dispatch({
    selection: EditorSelection.cursor(anchor),
    scrollIntoView: true,
  });
  view.focus();
}

export function dictateTextTransaction(state: EditorState, insertText: string): TransactionSpec {
  const anchor = getDictationAnchor(state);
  const newAnchor = anchor + insertText.length;
  const mainSel = state.selection.main;
  const isCursorAtAnchor = mainSel.empty && mainSel.head === anchor;

  const spec: TransactionSpec = {
    changes: { from: anchor, insert: insertText },
    annotations: [Transaction.userEvent.of("input.dictation")],
    effects: [setAnchorEffect.of(newAnchor)],
  };

  if (isCursorAtAnchor) {
    spec.selection = EditorSelection.cursor(newAnchor);
  }

  return spec;
}

// ─── Visual Anchor Marker Decoration ──────────────────────────────────────────

class AnchorMarkerWidget extends WidgetType {
  toDOM(): HTMLElement {
    const span = document.createElement("span");
    span.className = "cm-dictation-anchor";
    span.setAttribute("aria-hidden", "true");
    span.title = "Dictation anchor: speech inserts here";
    span.textContent = "⚓";
    return span;
  }

  eq(_other: AnchorMarkerWidget): boolean {
    return true;
  }

  ignoreEvent(): boolean {
    return true;
  }
}

export const dictationAnchorDecoration = StateField.define<DecorationSet>({
  create(state: EditorState): DecorationSet {
    const pos = getDictationAnchor(state);
    return Decoration.set([
      Decoration.widget({
        widget: new AnchorMarkerWidget(),
        side: 1,
      }).range(pos),
    ]);
  },
  update(_decorations: DecorationSet, tr: Transaction): DecorationSet {
    const pos = getDictationAnchor(tr.state);
    return Decoration.set([
      Decoration.widget({
        widget: new AnchorMarkerWidget(),
        side: 1,
      }).range(pos),
    ]);
  },
  provide: (f) => EditorView.decorations.from(f),
});
