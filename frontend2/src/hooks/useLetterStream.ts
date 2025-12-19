// File: src/hooks/useLetterStream.ts

import { useEffect, useState, useRef, useCallback } from "react";

export type LetterModelEvent = {
  timestamp: number;
  distribution: { [char: string]: number };
  predicted: string;
  confidence: number;
};

export type LetterEvent = {
  char: string;
  confidence: number;
  alternatives: Array<{ char: string; confidence: number }>;
  timestamp: number;
};

type UseLetterStreamProps = {
  onLetter: (ev: LetterEvent) => void;
  modelEvents$: AsyncIterable<LetterModelEvent> | (() => Promise<LetterModelEvent>[]); 
  /** 
   * Either you pass a stream of model events or a callback returning events.
   */
  debounceMs?: number;
  minConfidence?: number;
};

export function useLetterStream({
  modelEvents$,
  onLetter,
  debounceMs = 200,
  minConfidence = 0.5,
}: UseLetterStreamProps) {
  const bufferRef = useRef<LetterModelEvent[]>([]);

  useEffect(() => {
    let mounted = true;
    async function listen() {
      if (typeof modelEvents$ === "function") {
        // simple polling
        while (mounted) {
          const ev = await modelEvents$();
          if (!mounted) break;
          bufferRef.current.push(ev);
          await new Promise(r => setTimeout(r, 10));
        }
      } else {
        for await (const ev of modelEvents$) {
          if (!mounted) break;
          bufferRef.current.push(ev);
        }
      }
    }
    listen();

    const handler = setInterval(() => {
      const buf = bufferRef.current;
      if (buf.length === 0) return;
      // find highest confidence in buffer
      const best = buf.reduce((acc, e) => (e.confidence > acc.confidence ? e : acc), buf[0]);
      if (best.confidence >= minConfidence) {
        const sortedAlts = [...best.distribution]
          .sort((a, b) => b[1] - a[1])
          .map(([char, conf]) => ({ char, confidence: conf }));
        onLetter({
          char: best.predicted,
          confidence: best.confidence,
          alternatives: sortedAlts.slice(1, 4),
          timestamp: best.timestamp,
        });
        bufferRef.current = [];
      }
      else {
        // if buffer old, drop
        bufferRef.current = [];
      }
    }, debounceMs);

    return () => {
      mounted = false;
      clearInterval(handler);
    };
  }, [modelEvents$, onLetter, debounceMs, minConfidence]);
}
