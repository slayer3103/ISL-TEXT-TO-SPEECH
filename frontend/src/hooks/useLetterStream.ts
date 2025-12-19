/**
 * useLetterStreamAdapter
 * Normalizes raw model distributions, stabilizes over a small window,
 * and emits LetterEvent via onLetter callback. Also provides functions
 * to push raw model events, lock emission (to avoid race during Accept),
 * and access prev/next history items.
 */

import { useEffect, useRef, useState, useCallback } from "react";

export type LetterModelEvent = {
  timestamp: number;
  distribution: { [char: string]: number }; // raw scores or probs
  predicted: string;                         // argmax from model
  confidence: number;                        // raw as emitted
};

export type LetterEvent = {
  char: string;
  confidence: number; // normalized 0..1
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
  const values = Object.values(dist);
  if (values.length === 0) return dist;
  const max = Math.max(...values);
  let normalized: { [k: string]: number } = {};
  if (max > 1.1) { // looks like percentages (0-100)
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
  const bufRef = useRef<Array<LetterModelEvent>>([]);
  const emitLockRef = useRef(false);
  const lastEmittedRef = useRef<LetterEvent | null>(null);
  const [history, setHistory] = useState<LetterEvent[]>([]);
  const historyRef = useRef<LetterEvent[]>([]);
  historyRef.current = history;

  const pushModelEvent = useCallback((raw: LetterModelEvent) => {
    // normalize distribution
    raw.distribution = normalizeDistribution(raw.distribution);
    bufRef.current.push(raw);
    if (bufRef.current.length > windowSize) bufRef.current.shift();
  }, [windowSize]);

  const lockEmission = useCallback((locked: boolean) => {
    emitLockRef.current = locked;
  }, []);

  const getPrev = useCallback(() => {
    const h = historyRef.current;
    if (h.length <= 1) return null;
    return h[h.length - 2];
  }, []);

  const getNext = useCallback(() => {
    const h = historyRef.current;
    return h[h.length - 1] || null;
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      if (emitLockRef.current) return;
      const buf = bufRef.current;
      if (!buf.length) return;

      const agg: { [k: string]: number } = {};
      for (const e of buf) {
        for (const k in e.distribution) {
          agg[k] = (agg[k] || 0) + (e.distribution[k] || 0);
        }
      }
      const denom = buf.length || 1;
      for (const k in agg) agg[k] = agg[k] / denom;

      // normalize again
      const norm = normalizeDistribution(agg);
      const entries = Object.entries(norm).sort((a, b) => b[1] - a[1]);
      if (entries.length === 0) return;
      const [predChar, predProb] = entries[0] as [string, number];

      if (predProb < minConfidence) {
        // do not emit low-confidence events
        return;
      }

      const alternatives = entries.slice(1, 5).map(([c, p]) => ({ char: c, confidence: p }));
      const ev: LetterEvent = { char: predChar, confidence: predProb, alternatives, timestamp: Date.now() };
      lastEmittedRef.current = ev;
      setHistory(h => {
        const nh = [...h, ev].slice(-16);
        return nh;
      });
      onLetter(ev);
      bufRef.current = [];
    }, debounceMs);

    return () => clearInterval(id);
  }, [onLetter, debounceMs, minConfidence]);

  return { pushModelEvent, lockEmission, getPrev, getNext, history };
}
