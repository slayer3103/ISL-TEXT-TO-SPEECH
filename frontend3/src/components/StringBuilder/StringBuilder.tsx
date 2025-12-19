import React, { useCallback, useEffect, useState, forwardRef, useImperativeHandle } from "react";
import { useUndoStack } from "../../hooks/useUndoStack";
import { SuggestionEngine } from "../../lib/suggestions/suggestions";
import { LetterEvent } from "../../hooks/useLetterStream";

type Props = {
  onCommittedChange?: (s: string) => void;
  suggestionEngine: SuggestionEngine;
  pushModelEvent: (ev: any) => void;
  lockEmission: (locked: boolean) => void;
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

  const [historyIndex, setHistoryIndex] = useState<number | null>(null);
  const currentPrefix = committed.split(" ").pop() || "";

  // Keep pending synced with latest history item
  useEffect(() => {
    if (history && history.length) {
      setHistoryIndex(history.length - 1);
      setPending(history[history.length - 1]);
    } else {
      setHistoryIndex(null);
      setPending(null);
    }
  }, [history]);

  useEffect(() => {
    if (typeof onCommittedChange === "function") onCommittedChange(committed);
  }, [committed, onCommittedChange]);

  useEffect(() => {
    if (!pending) { setSuggestions([]); return; }
    const distMap: { [k: string]: number } = { [pending.char]: pending.confidence };
    for (const a of pending.alternatives || []) distMap[a.char] = a.confidence;
    const top = suggestionEngine.getTopSuggestions(currentPrefix, distMap, 5);
    setSuggestions(top.map(t => t.word));
  }, [pending, currentPrefix, suggestionEngine]);

  const acceptLetter = useCallback(() => {
    if (!pending) return;
    lockEmission(true);
    push({ type: "acceptLetter", payload: { text: pending.char, length: 1 } });
    setCommitted(prev => prev + pending.char);
    setPending(null);
    setTimeout(() => lockEmission(false), 180);
  }, [pending, lockEmission, push]);

  const acceptWord = useCallback((word: string) => {
    lockEmission(true);
    const suffix = word.slice(currentPrefix.length);
    push({ type: "acceptWord", payload: { text: suffix + " ", length: suffix.length + 1 } });
    setCommitted(prev => prev + suffix + " ");
    setPending(null);
    setSuggestions([]);
    setTimeout(() => lockEmission(false), 200);
  }, [currentPrefix, lockEmission, push]);

  const space = useCallback(() => {
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
    setCommitted(prev => prev.slice(0, prev.length - op.payload.length));
  }, [undo]);

  const handlePrev = useCallback(() => {
    if (!history || history.length === 0 || historyIndex === null) return;
    const ni = Math.max(0, (historyIndex || 0) - 1);
    setHistoryIndex(ni);
    setPending(history[ni]);
  }, [history, historyIndex]);

  const handleNext = useCallback(() => {
    if (!history || history.length === 0 || historyIndex === null) return;
    const ni = Math.min(history.length - 1, (historyIndex || 0) + 1);
    setHistoryIndex(ni);
    setPending(history[ni]);
  }, [history, historyIndex]);

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

  // Exposed: insert arbitrary text (used for inserting the last model output)
  const insertLastOutput = useCallback((txt: string) => {
    if (!txt) return;
    // treat it as a word and record for undo
    push({ type: "acceptWord", payload: { text: txt + " ", length: txt.length + 1 } });
    setCommitted(prev => prev + txt + " ");
  }, [push]);

  // expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    space: () => { space(); },
    deleteLast: () => { deleteLast(); },
    undo: () => { handleUndo(); },
    finalize: () => { finalize(); },
    acceptLetter: () => { acceptLetter(); },
    prev: () => { handlePrev(); },
    next: () => { handleNext(); },
    insertLastOutput: (t: string) => { insertLastOutput(t); }
  }), [space, deleteLast, handleUndo, finalize, acceptLetter, handlePrev, handleNext, insertLastOutput]);

  return (
    <div className="string-builder">
      <div className="committed-area" style={{ whiteSpace: "pre-wrap", padding: 8, border: "1px solid #f1f1f1", borderRadius: 6 }}>
        {committed || <span style={{ color: "#9aa0a6" }}>Letters will appear here...</span>}<span className="caret">|</span>
      </div>

      {/* left-side now only shows the committed string. pending / suggestions / confirmation are handled in the Controls card */}
    </div>
  );
});

export default StringBuilder;
