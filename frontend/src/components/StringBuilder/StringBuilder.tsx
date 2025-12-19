// src/components/StringBuilder/StringBuilder.tsx
import React, { useCallback, useEffect, useState, forwardRef, useImperativeHandle } from "react";
import { useUndoStack } from "../../hooks/useUndoStack";
import { SuggestionEngine } from "../../lib/suggestions/suggestions";
import { LetterEvent } from "../../hooks/useLetterStream";

type Props = {
  onCommittedChange?: (s: string) => void;
  suggestionEngine: SuggestionEngine;
  pushModelEvent?: (ev: any) => void;
  lockEmission?: (locked: boolean) => void;
  getPrev?: () => LetterEvent | null;
  getNext?: () => LetterEvent | null;
  history?: LetterEvent[];
};

export const StringBuilder = forwardRef(function StringBuilderInner(props: Props, ref: any) {
  const { suggestionEngine, pushModelEvent, lockEmission, getPrev, getNext, history, onCommittedChange } = props;

  const [committed, setCommitted] = useState<string>("");
  const [pending, setPending] = useState<LetterEvent | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const { push, undo } = useUndoStack();

  // word-level cursor 0..N (between words). cursor > 0 means word at index cursor-1 is selected
  const [wordCursor, setWordCursor] = useState<number>(0);

  // keep pending synced with history last
  useEffect(() => {
    if (history && history.length) {
      setPending(history[history.length - 1]);
    } else {
      setPending(null);
    }
  }, [history]);

  // notify parent when committed changes
  useEffect(() => {
    if (typeof onCommittedChange === "function") onCommittedChange(committed);
    // clamp cursor to new word count
    const words = committed.trim() ? committed.trim().split(/\s+/) : [];
    setWordCursor((c) => Math.max(0, Math.min(c, words.length)));
  }, [committed, onCommittedChange]);

  // suggestions update from pending
    // suggestions update from pending
  const currentPrefix = committed.split(" ").pop() || "";
  useEffect(() => {
    if (!pending) { setSuggestions([]); return; }
    const distMap: { [k: string]: number } = { [pending.char]: pending.confidence };
    for (const a of pending.alternatives || []) distMap[a.char] = a.confidence;

    // DEBUG: log pending and prefix
    try {
      console.debug("[StringBuilder] pending:", pending, "currentPrefix:", currentPrefix);
    } catch (e) {}

    const top = suggestionEngine.getTopSuggestions(currentPrefix, distMap, 5);

    // DEBUG: log what engine returned
    try {
      console.debug("[StringBuilder] suggestionEngine returned:", top);
    } catch (e) {}

    // ensure compatibility: top may be string[] or {word,...} depending on engine
    if (top && top.length && typeof top[0] === "string") {
      setSuggestions(top as string[]);
    } else if (top && top.length && typeof (top[0] as any).word === "string") {
      setSuggestions((top as any[]).map((t) => (t.word || String(t)).toString()));
    } else {
      setSuggestions([]);
    }
  }, [pending, currentPrefix, suggestionEngine]);


  // Basic operations (undo stack usage)
  const space = useCallback(() => {
    // insert a space at end (caret logic could be extended to insert at cursor location)
    push({ type: "acceptLetter", payload: { text: " ", length: 1 } });
    setCommitted(prev => prev + " ");
  }, [push]);

  const deleteLast = useCallback(() => {
    if (!committed.length) return;
    push({ type: "delete", payload: { text: committed.slice(-1), length: 1 } });
    setCommitted(prev => prev.slice(0, -1));
  }, [committed, push]);

  const handleUndo = useCallback(() => {
    const op = undo();
    if (!op) return;
    // rollback by removing op.payload.length chars from end
    setCommitted(prev => prev.slice(0, Math.max(0, prev.length - (op.payload?.length || 0))));
  }, [undo]);

  // Word-level caret helpers
  const prevWord = useCallback(() => {
    setWordCursor(c => Math.max(0, c - 1));
  }, []);

  const nextWord = useCallback(() => {
    const words = committed.trim() ? committed.trim().split(/\s+/) : [];
    setWordCursor(c => Math.min(words.length, c + 1));
  }, [committed]);

  // Insert text at cursor without trailing space.
  // Fixed: if committed currently has an explicit trailing space before insertion,
  // treat a single-letter insertion as a new word (do NOT append to previous word).
  const insertAtCursor = useCallback((txt: string) => {
    if (!txt) return;
    const token = String(txt).replace(/\s+$/u, ""); // strip trailing spaces from caller token
    if (!token) return;

    const hadTrailing = /\s$/u.test(committed); // preserve explicit trailing space if user had it

    const raw = committed.trim();
    const words = raw ? raw.split(/\s+/) : [];
    const before = words.slice(0, wordCursor);
    const after = words.slice(wordCursor);

    // Single-char special-case: **only append to last word** when there's no explicit trailing space.
    // If user had an explicit trailing space (they pressed Space), treat incoming single char as a new word.
    if (token.length === 1 && wordCursor === words.length && !hadTrailing) {
      // append single-char to last word (letter-mode)
      if (words.length === 0) {
        const newCommitted = token + (hadTrailing ? " " : "");
        push({ type: "acceptLetter", payload: { text: token, length: 1 } });
        setCommitted(newCommitted);
        setWordCursor(1);
        return;
      } else {
        const last = words[words.length - 1] + token;
        const newWords = [...words.slice(0, -1), last];
        const newCommitted = newWords.join(" ") + (hadTrailing ? " " : "");
        push({ type: "acceptLetter", payload: { text: token, length: 1 } });
        setCommitted(newCommitted);
        setWordCursor(newWords.length);
        return;
      }
    }

    // Default: insert as separate token/word (no trailing space appended)
    const newWords = [...before, token, ...after];
    const newCommitted = newWords.join(" ") + (hadTrailing ? " " : "");
    push({ type: "acceptWord", payload: { text: token, length: token.length } });
    setCommitted(newCommitted);
    setWordCursor(before.length + 1);
  }, [committed, wordCursor, push]);

  // Replace selected word (cursor > 0 selects word at cursor-1)
  const replaceSelectedWord = useCallback((txt: string) => {
    const token = String(txt).replace(/\s+$/u, "");
    if (!token) return;

    const raw = committed.trim();
    const words = raw ? raw.split(/\s+/) : [];
    if (wordCursor === 0) {
      insertAtCursor(token);
      return;
    }
    const idx = wordCursor - 1;
    if (idx < 0 || idx >= words.length) {
      insertAtCursor(token);
      return;
    }
    const before = words.slice(0, idx);
    const after = words.slice(idx + 1);
    const newWords = [...before, token, ...after];
    const newCommitted = newWords.join(" ") + (/\s$/u.test(committed) ? " " : "");
    push({ type: "replace", payload: { old: words[idx], text: token, length: token.length } });
    setCommitted(newCommitted);
    setWordCursor(before.length + 1);
  }, [committed, wordCursor, insertAtCursor, push]);

  // Accept helpers (kept for programmatic use if other code calls them)
  const acceptLetter = useCallback(() => {
    if (!pending) return;
    lockEmission && lockEmission(true);
    push({ type: "acceptLetter", payload: { text: pending.char, length: 1 } });
    setCommitted(prev => prev + pending.char);
    setPending(null);
    setTimeout(() => lockEmission && lockEmission(false), 180);
  }, [pending, lockEmission, push]);

  const acceptWord = useCallback((word: string) => {
    const suffix = word.slice(currentPrefix.length);
    lockEmission && lockEmission(true);
    // do NOT add trailing space automatically
    push({ type: "acceptWord", payload: { text: suffix, length: suffix.length } });
    setCommitted(prev => {
      if (!prev) return suffix;
      const hadTrailing = /\s$/u.test(prev);
      const prefix = prev.replace(/\s+$/u, "");
      return prefix + (prefix ? " " : "") + suffix + (hadTrailing ? " " : "");
    });
    setPending(null);
    setSuggestions([]);
    setTimeout(() => lockEmission && lockEmission(false), 200);
  }, [currentPrefix, lockEmission, push]);

  // finalize with TTS
  const finalize = useCallback(() => {
    const finalText = committed.trim();
    if (!finalText) return;
    if ("speechSynthesis" in window) {
      const ut = new SpeechSynthesisUtterance(finalText);
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(ut);
    } else {
      console.log("TTS not available", finalText);
    }
  }, [committed]);

  // Expose imperative methods
  useImperativeHandle(ref, () => ({
    space,
    deleteLast,
    undo: handleUndo,
    finalize,
    acceptLetter,
    acceptWord,
    prev: prevWord,
    next: nextWord,
    insertLastOutput: (t: string) => insertAtCursor(t), // alias
    replaceAtCursor: (t: string) => replaceSelectedWord(t),
    getCommitted: () => committed,
    setCommittedText: (s: string) => setCommitted(s),
    getWordCursor: () => wordCursor,
    setWordCursor: (n: number) => setWordCursor(n)
  }), [space, deleteLast, handleUndo, finalize, acceptLetter, acceptWord, prevWord, nextWord, insertAtCursor, replaceSelectedWord, committed, wordCursor]);

  // Render committed text with selected-word highlight
  const renderCommittedWithHighlight = () => {
    const rawCommitted = committed;
    if (!rawCommitted || rawCommitted.trim() === "") {
      if (/\s/u.test(rawCommitted)) {
        return <span style={{ color: "#9aa0a6" }}>&nbsp;</span>;
      }
      return <span style={{ color: "#9aa0a6" }}>Letters will appear here...</span>;
    }

    const words = rawCommitted.trim().split(/\s+/);
    return (
      <span>
        {words.map((w, i) => {
          const isSelected = wordCursor === i + 1;
          return (
            <span key={i} style={{ marginRight: 6 }}>
              <span style={isSelected ? { background: "#fff3b0", padding: "2px 6px", borderRadius: 4, fontWeight: 700 } : {}}>
                {w}
              </span>
              {i < words.length - 1 ? " " : ""}
            </span>
          );
        })}
        {(/\s$/u.test(rawCommitted) || wordCursor === (rawCommitted.trim().split(/\s+/).length)) ? <span style={{ display: "inline-block", width: 8, marginLeft: 4 }}>▌</span> : null}
      </span>
    );
  };

  return (
    <div className="string-builder">
      <div className="committed-area" style={{ whiteSpace: "pre-wrap", padding: 8, border: "1px solid #f1f1f1", borderRadius: 6 }}>
        {renderCommittedWithHighlight()}
      </div>
      {/* Controls live elsewhere (Demo) */}
    </div>
  );
});

export default StringBuilder;
