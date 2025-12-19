// src/hooks/useLetterStream.ts
import { useEffect, useRef, useState, useCallback } from "react";

export type LetterModelEvent = {
  timestamp: number;
  distribution: { [char: string]: number }; // raw scores or probs
  predicted: string;                         // argmax from model
  confidence: number;                        // raw as emitted
};

export type LetterEvent = {
  char: string;
  confidence: number;                         // normalized 0..1
  alternatives: Array<{ char: string; confidence: number }>;
  timestamp: number;
};

type Props = {
  onLetter: (ev: LetterEvent) => void;
  debounceMs?: number;
  windowSize?: number;    // number of recent events to average
  minConfidence?: number; // min emission prob to emit
};

function normalizeDistribution(dist: { [k: string]: number }) {
  // If the numbers look like percentages (values > 1), divide by 100
  const values = Object.values(dist);
  if (values.length === 0) return dist;

  const max = Math.max(...values);
  let normalized: { [k: string]: number } = {};
  if (max > 1.1) { // e.g., raw percentages like 85
    for (const k in dist) normalized[k] = dist[k] / 100;
  } else {
    const sum = values.reduce((s, v) => s + v, 0) || 1;
    for (const k in dist) normalized[k] = dist[k] / sum;
  }
  return normalized;
}

export function useLetterStreamAdapter({
  onLetter,
  debounceMs = 150,
  windowSize = 4,
  minConfidence = 0.15,
}: Props) {
  // internal circular buffer of raw model events
  const bufRef = useRef<Array<LetterModelEvent>>([]);
  const emitLockRef = useRef(false); // used to lock during accept
  const lastEmittedRef = useRef<LetterEvent | null>(null);

  // public functions for UI to navigate prev/next or lock
  const [history, setHistory] = useState<LetterEvent[]>([]); // last emitted events
  const historyRef = useRef<LetterEvent[]>([]);
  historyRef.current = history;

  // consumer should call this to push raw model events
  const pushModelEvent = useCallback((raw: LetterModelEvent) => {
    // normalize distribution shape
    raw.distribution = normalizeDistribution(raw.distribution);
    bufRef.current.push(raw);
    // keep buffer bounded
    if (bufRef.current.length > windowSize) bufRef.current.shift();
  }, [windowSize]);

  // call this to temporarily lock (used when user Accepts)
  const lockEmission = useCallback((locked: boolean) => {
    emitLockRef.current = locked;
  }, []);

  // expose prev/next navigation
  const getPrev = useCallback(() => {
    const h = historyRef.current;
    if (h.length <= 1) return null;
    return h[h.length - 2];
  }, []);

  const getNext = useCallback(() => {
    const h = historyRef.current;
    // we don't keep a future buffer here; next would be last emitted if not at end
    return h[h.length - 1] || null;
  }, []);

  // debounce/aggregate timer
  useEffect(() => {
    const id = setInterval(() => {
      if (emitLockRef.current) return; // locked due to Accept action

      const buf = bufRef.current;
      if (!buf.length) return;

      // average distributions across buffer
      const agg: { [k: string]: number } = {};
      for (const e of buf) {
        for (const k in e.distribution) {
          agg[k] = (agg[k] || 0) + (e.distribution[k] || 0);
        }
      }
      const denom = buf.length || 1;
      for (const k in agg) agg[k] = agg[k] / denom;

      // normalize agg again to be safe
      const norm = normalizeDistribution(agg);

      // compute top predicted char from norm
      const sorted = Object.entries(norm).sort((a, b) => b[1] - a[1]);
      const [predChar, predProb] = sorted[0] ?? ["", 0];

      // if confidence below minConfidence, do not emit (or emit blank weak)
      if (predProb < minConfidence) {
        // Optionally emit a "no-letter" event; we'll skip for now
        return;
      }

      const alternatives = sorted.slice(1, 5).map(([c, p]) => ({ char: c, confidence: p }));
      const ev: LetterEvent = { char: predChar, confidence: predProb, alternatives, timestamp: Date.now() };

      // emit to consumer and append to history
      lastEmittedRef.current = ev;
      setHistory(h => {
        const nh = [...h, ev].slice(-8); // keep last 8 entries
        return nh;
      });
      onLetter(ev);

      // clear buffer after emitting
      bufRef.current = [];
    }, debounceMs);

    return () => clearInterval(id);
  }, [onLetter, debounceMs, minConfidence]);

  return { pushModelEvent, lockEmission, getPrev, getNext, history };
}
