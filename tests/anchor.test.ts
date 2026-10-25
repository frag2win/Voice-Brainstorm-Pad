import { describe, it, expect } from "vitest";
import { EditorState, Transaction, EditorSelection } from "@codemirror/state";
import {
  dictationAnchorField,
  setAnchorEffect,
  getDictationAnchor,
  dictateTextTransaction,
} from "../src/renderer/anchor";

describe("dictationAnchorField - CodeMirror State Field & Position Mapping", () => {
  it("initializes anchor at the end of the document", () => {
    const text = "Hello world";
    const state = EditorState.create({
      doc: text,
      extensions: [dictationAnchorField],
    });
    expect(getDictationAnchor(state)).toBe(text.length);
  });

  it("shifts anchor forward when typing before the anchor", () => {
    const state = EditorState.create({
      doc: "world",
      extensions: [dictationAnchorField],
    });
    // Anchor starts at 5
    expect(getDictationAnchor(state)).toBe(5);

    // Insert "Hello " at position 0
    const tr = state.update({
      changes: { from: 0, insert: "Hello " },
    });
    // Anchor should shift from 5 to 5 + 6 = 11
    expect(getDictationAnchor(tr.state)).toBe(11);
  });

  it("does not shift anchor when typing after the anchor", () => {
    const state = EditorState.create({
      doc: "Hello world",
      extensions: [dictationAnchorField],
    });
    // Move anchor to 5 (after "Hello")
    const s1 = state.update({
      effects: [setAnchorEffect.of(5)],
    }).state;
    expect(getDictationAnchor(s1)).toBe(5);

    // Type text at position 11 (after anchor)
    const s2 = s1.update({
      changes: { from: 11, insert: "!" },
    }).state;
    expect(getDictationAnchor(s2)).toBe(5);
  });

  it("shifts anchor after typed text when typing exactly at the anchor (forward association)", () => {
    const state = EditorState.create({
      doc: "Hello",
      extensions: [dictationAnchorField],
    });
    expect(getDictationAnchor(state)).toBe(5);

    // User types " there" exactly at position 5
    const tr = state.update({
      changes: { from: 5, insert: " there" },
    });
    // Because assoc = 1, anchor should stay after the typed text (pos 11)
    expect(getDictationAnchor(tr.state)).toBe(11);
  });

  it("shifts anchor backward when deleting text before the anchor", () => {
    const state = EditorState.create({
      doc: "Hello beautiful world",
      extensions: [dictationAnchorField],
    });
    // Anchor at 21
    expect(getDictationAnchor(state)).toBe(21);

    // Delete "beautiful " (10 chars from 6 to 16)
    const tr = state.update({
      changes: { from: 6, to: 16 },
    });
    // Anchor should shift from 21 to 11
    expect(getDictationAnchor(tr.state)).toBe(11);
    expect(tr.state.doc.toString()).toBe("Hello world");
  });

  it("collapses anchor to the deletion point when text containing the anchor is deleted", () => {
    const state = EditorState.create({
      doc: "First sentence. Second sentence.",
      extensions: [dictationAnchorField],
    });
    // Set anchor at pos 15 (end of "First sentence.")
    const s1 = state.update({
      effects: [setAnchorEffect.of(15)],
    }).state;
    expect(getDictationAnchor(s1)).toBe(15);

    // Delete from pos 10 to 25 (encompassing the anchor)
    const s2 = s1.update({
      changes: { from: 10, to: 25 },
    });
    // Anchor should collapse to the deletion start point (10), never to 0
    expect(getDictationAnchor(s2.state)).toBe(10);
  });

  it("allows explicitly moving the anchor using setAnchorEffect", () => {
    const state = EditorState.create({
      doc: "Line one\nLine two\nLine three",
      extensions: [dictationAnchorField],
    });
    const s1 = state.update({
      effects: [setAnchorEffect.of(9)],
    }).state;
    expect(getDictationAnchor(s1)).toBe(9);
  });

  it("dictates while typing corrections three paragraphs above without disturbing typed text", () => {
    const p1 = "First paragraph with an error.\n\n";
    const p2 = "Second paragraph.\n\n";
    const p3 = "Third paragraph.";
    const state = EditorState.create({
      doc: p1 + p2 + p3,
      extensions: [dictationAnchorField],
    });
    const anchorInit = getDictationAnchor(state);
    expect(anchorInit).toBe((p1 + p2 + p3).length);

    // User types correction three paragraphs above (replaces "an error" with "a correction" in p1)
    const errIdx = p1.indexOf("an error");
    const correction = "a fixed sentence";
    const s1 = state.update({
      changes: { from: errIdx, to: errIdx + "an error".length, insert: correction },
    }).state;

    // Shift in length: correction.length - "an error".length
    const shift = correction.length - "an error".length;
    expect(getDictationAnchor(s1)).toBe(anchorInit + shift);

    // Now incoming dictation chunk arrives at the anchor
    const s2 = s1.update(dictateTextTransaction(s1, " New thought.")).state;

    // Confirm correction in p1 remains intact
    expect(s2.doc.toString().startsWith("First paragraph with a fixed sentence.\n\n")).toBe(true);
    // Confirm dictation landed at the very end of p3
    expect(s2.doc.toString().endsWith("Third paragraph. New thought.")).toBe(true);
  });

  it("supports undoing dictated text chunk via CodeMirror history", async () => {
    const { history, undo } = await import("@codemirror/commands");
    const state = EditorState.create({
      doc: "Start: ",
      extensions: [history(), dictationAnchorField],
    });

    const s1 = state.update(dictateTextTransaction(state, "chunk one")).state;
    expect(s1.doc.toString()).toBe("Start: chunk one");

    // Perform undo
    let undone = false;
    undo({
      state: s1,
      dispatch: (tr) => {
        undone = true;
        expect(tr.state.doc.toString()).toBe("Start: ");
      },
    });
    expect(undone).toBe(true);
  });

  it("caret at anchor moves to after dictated text, and typing one more word lands after it", () => {
    const text = "Initial text";
    let state = EditorState.create({
      doc: text,
      selection: EditorSelection.cursor(text.length),
      extensions: [dictationAnchorField],
    });
    expect(state.selection.main.head).toBe(text.length);

    // Dictate " more speech"
    const dictated = " more speech";
    const tr = state.update(dictateTextTransaction(state, dictated));
    state = tr.state;

    // Caret must have followed to the end of inserted text
    expect(state.selection.main.empty).toBe(true);
    expect(state.selection.main.head).toBe(text.length + dictated.length);
    expect(getDictationAnchor(state)).toBe(text.length + dictated.length);

    // Typing one more word lands after it
    const typedWord = " typed";
    const tr2 = state.update({
      changes: { from: state.selection.main.head, insert: typedWord },
      selection: EditorSelection.cursor(state.selection.main.head + typedWord.length),
    });
    state = tr2.state;
    expect(state.doc.toString()).toBe("Initial text more speech typed");
  });

  it("caret elsewhere (position 0) remains unchanged and mapped correctly", () => {
    const text = "Initial text";
    let state = EditorState.create({
      doc: text,
      selection: EditorSelection.cursor(0),
      extensions: [dictationAnchorField],
    });
    expect(getDictationAnchor(state)).toBe(12);

    const tr = state.update(dictateTextTransaction(state, " appended"));
    state = tr.state;

    // Caret at pos 0 must remain at pos 0
    expect(state.selection.main.empty).toBe(true);
    expect(state.selection.main.head).toBe(0);
    expect(state.doc.toString()).toBe("Initial text appended");
  });

  it("non-empty selection range is preserved and mapped correctly", () => {
    const text = "Selected text here";
    let state = EditorState.create({
      doc: text,
      selection: EditorSelection.range(0, 8),
      extensions: [dictationAnchorField],
    });
    expect(state.selection.main.empty).toBe(false);

    const tr = state.update(dictateTextTransaction(state, " more"));
    state = tr.state;

    // Selection range must still be 0 to 8
    expect(state.selection.main.empty).toBe(false);
    expect(state.selection.main.from).toBe(0);
    expect(state.selection.main.to).toBe(8);
    expect(state.sliceDoc(state.selection.main.from, state.selection.main.to)).toBe("Selected");
  });
});
