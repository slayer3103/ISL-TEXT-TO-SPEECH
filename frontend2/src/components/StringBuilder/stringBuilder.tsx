// File: src/components/StringBuilder/StringBuilder.tsx

import React, { useState, useCallback, useEffect } from "react";
import { SuggestionPills } from "./SuggestionPills";
import { LetterConfirm } from "./LetterConfirm";
import { Controls } from "./Controls";
import { useUndoStack } from "../../hooks/useUndoStack";
import { SuggestionEngine } from "../../lib/suggestions/suggestions";

type LetterAlt = { char: string; confidence: number };

type Props = {
  // optional external controlled value (for Demo integration)
  value?: string;
  onChange?: (s: string) => void;

  // external ref to expose handlers
  controlRef?: React.MutableRefObject<any>;

  // incoming model event informing the pending character
  letterEvent: {
    char: string;
    confidence: number;
    alternatives: Array<LetterAlt>;
    timestamp: number;
  } | null;

  // suggestion engine instance (required)
  suggestionEngine: SuggestionEngine;
};

export const StringBuilder: React.FC<Props> = ({
  value,
  onChange,
  controlRef,
  letterEvent,
  suggestionEngine,
}) => {
  // if parent passes `value`, treat component as controlled for committed text
  const [committed, setCommitted] = useState<string>(value ?? "");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const { undoStack, push, undo, reset } = useUndoStack();

  // keep committed in sync when parent controls value
  useEffect(() => {
    if (typeof value === "string" && value !== committed) {
      setCommitted(value);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  // call parent onChange whenever committed changes (if onChange provided)
  useEffect(() => {
    if (onChange) onChange(committed);
  }, [committed, onChange]);

  const currentPrefix = committed.split(" ").pop() || "";

  const updateSuggestions = useCallback(() => {
    if (!letterEvent) {
      setSuggestions([]);
      return;
    }
    const distMap: { [k: string]: number } = {
      [letterEvent.char]: letterEvent.confidence,
      ...Object.fromEntries(letterEvent.alternatives.map((a) => [a.char, a.confidence])),
    };
    const top = suggestionEngine.getTopSuggestions(currentPrefix, distMap, 5);
    setSuggestions(top.map((t) => t.word));
  }, [letterEvent, suggestionEngine, currentPrefix]);

  useEffect(() => {
    updateSuggestions();
  }, [letterEvent, currentPrefix, updateSuggestions]);

  const acceptLetter = useCallback(() => {
    if (!letterEvent) return;
    push({ type: "acceptLetter", payload: { text: letterEvent.char, length: 1 } });
    setCommitted((prev) => prev + letterEvent.char);
  }, [letterEvent, push]);

  const acceptWord = useCallback(
    (word: string) => {
      const suffix = word.slice(currentPrefix.length);
      push({ type: "acceptWord", payload: { text: suffix + " ", length: suffix.length + 1 } });
      setCommitted((prev) => prev + suffix + " ");
      setSuggestions([]);
    },
    [currentPrefix, push]
  );

  const space = useCallback(() => {
    push({ type: "acceptLetter", payload: { text: " ", length: 1 } });
    setCommitted((prev) => prev + " ");
    setSuggestions([]);
  }, [push]);

  const deleteLast = useCallback(() => {
    if (committed.length === 0) return;
    push({ type: "delete", payload: { text: committed.slice(-1), length: 1 } });
    setCommitted((prev) => prev.slice(0, -1));
  }, [committed, push]);

  const handleUndo = useCallback(() => {
    const op = undo();
    if (!op) return;
    setCommitted((prev) => prev.slice(0, prev.length - op.payload.length));
  }, [undo]);

  const finalize = useCallback(() => {
    // call TTS or send committed.trim() to backend here if needed
    console.log("Final string:", committed.trim());
    // keep as-is or clear: setCommitted("");
  }, [committed]);

  const handlePrev = useCallback(() => {
    // optional: if you want prev/next navigation inside this component.
    // leave as no-op if external adapter manages history navigation.
    // Exposed for demo parity.
    console.log("StringBuilder.handlePrev called");
  }, []);

  const handleNext = useCallback(() => {
    console.log("StringBuilder.handleNext called");
  }, []);

  // expose control handlers to parent via controlRef (defensive)
  try {
    if (controlRef) {
      controlRef.current = {
        acceptLetter: typeof acceptLetter === "function" ? acceptLetter : undefined,
        acceptWord: typeof acceptWord === "function" ? acceptWord : undefined,
        space: typeof space === "function" ? space : undefined,
        deleteLast: typeof deleteLast === "function" ? deleteLast : undefined,
        handleUndo: typeof handleUndo === "function" ? handleUndo : undefined,
        finalize: typeof finalize === "function" ? finalize : undefined,
        handlePrev: typeof handlePrev === "function" ? handlePrev : undefined,
        handleNext: typeof handleNext === "function" ? handleNext : undefined,
        reset: typeof reset === "function" ? reset : undefined,
      };
    }
  } catch (err) {
    // defensive guard so render doesn't crash
    // console.warn("controlRef assign failed", err);
  }

  return (
    <div className="string-builder">
      <div className="committed-area">
        {committed}
        <span className="caret">|</span>
      </div>

      <LetterConfirm
        letterEvent={letterEvent}
        onAccept={acceptLetter}
        onReject={() => {
          /* maybe drop event or ask parent to advance */
        }}
      />

      <div style={{ marginTop: 8 }}>
        <SuggestionPills suggestions={suggestions} onAcceptWord={acceptWord} />
      </div>

      <div style={{ marginTop: 8 }}>
        <Controls onDelete={deleteLast} onUndo={handleUndo} onFinalize={finalize} />
      </div>
    </div>
  );
};

export default StringBuilder;
