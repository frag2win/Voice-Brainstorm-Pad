/**
 * Spacing and formatting rules for incoming dictation chunks (TRD 2.4).
 *
 * Rules:
 *  - Insert a leading space unless:
 *      * the anchor is at the start of the document (pos = 0)
 *      * the anchor is preceded by whitespace (space, tab, or newline)
 *      * the chunk text starts with punctuation that attaches directly (e.g. . , ! ? : ;)
 *  - After a pause longer than configured threshold (default 2.5s):
 *      * insert a paragraph break instead (\n\n, or \n if already preceded by \n)
 *      * unless at the very start of the document
 */

export interface SpacingOptions {
  pauseThresholdS?: number;
}

export interface FormatResult {
  insertText: string;
  hasLeadingBreak: boolean;
}

const ATTACHING_PUNCTUATION = new Set([".", ",", "!", "?", ":", ";", ")", "]", "}", "…"]);

export function calculateAudioPauseDuration(
  tStart?: number,
  lastChunkEnd?: number | null
): number {
  if (lastChunkEnd === null || lastChunkEnd === undefined || tStart === undefined) {
    return 0;
  }
  return Math.max(0, tStart - lastChunkEnd);
}

export function formatDictationText(
  docText: string,
  anchorPos: number,
  chunkText: string,
  pauseDurationS: number = 0,
  options: SpacingOptions = {}
): FormatResult {
  const cleanChunk = chunkText.trim();
  if (!cleanChunk) {
    return { insertText: "", hasLeadingBreak: false };
  }

  const threshold = options.pauseThresholdS ?? 2.5;
  const safeAnchor = Math.max(0, Math.min(anchorPos, docText.length));

  // If at document start, no leading space or newline
  if (safeAnchor === 0) {
    return { insertText: cleanChunk, hasLeadingBreak: false };
  }

  // Check if pause warrants a paragraph break
  if (pauseDurationS >= threshold) {
    let prefix = "\n\n";
    const sliceBefore = docText.slice(0, safeAnchor);
    if (sliceBefore.endsWith("\n\n")) {
      prefix = "";
    } else if (sliceBefore.endsWith("\n")) {
      prefix = "\n";
    }
    return { insertText: prefix + cleanChunk, hasLeadingBreak: true };
  }

  // Preceding char check
  const prevChar = docText[safeAnchor - 1];
  const firstChar = cleanChunk[0];

  // If previous char is whitespace, don't add space
  if (/\s/.test(prevChar)) {
    return { insertText: cleanChunk, hasLeadingBreak: false };
  }

  // If chunk starts with attaching punctuation, don't add space
  if (ATTACHING_PUNCTUATION.has(firstChar)) {
    return { insertText: cleanChunk, hasLeadingBreak: false };
  }

  // Default: prepend single space
  return { insertText: ` ${cleanChunk}`, hasLeadingBreak: false };
}

export function formatChunkFailedNotice(seq: number): string {
  return `Chunk ${seq} could not be transcribed (audio kept)`;
}

