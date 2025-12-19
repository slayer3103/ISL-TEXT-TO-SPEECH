import { useState, useCallback, useRef } from 'react';

export interface Distribution {
  [letter: string]: number;
}

export interface ModelEvent {
  distribution: Distribution;
  label: string;
  confidence: number;
  alternatives?: string[];
}

export interface LetterEvent {
  letter: string;
  confidence: number;
  distribution: Distribution;
  alternatives: string[];
}

interface LetterStreamAdapterConfig {
  bufferSize?: number;
  emissionThreshold?: number;
}

export function useLetterStreamAdapter(config: LetterStreamAdapterConfig = {}) {
  const { bufferSize = 5, emissionThreshold = 0.6 } = config;

  const [buffer, setBuffer] = useState<ModelEvent[]>([]);
  const [currentLetter, setCurrentLetter] = useState<LetterEvent | null>(null);
  const [history, setHistory] = useState<LetterEvent[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const isLockedRef = useRef(false);

  const normalizeDistribution = (dist: Distribution): Distribution => {
    const sum = Object.values(dist).reduce((acc, val) => acc + val, 0);
    if (sum === 0) return dist;
    
    const normalized: Distribution = {};
    for (const [key, val] of Object.entries(dist)) {
      normalized[key] = val / sum;
    }
    return normalized;
  };

  const averageDistributions = (events: ModelEvent[]): Distribution => {
    if (events.length === 0) return {};
    
    const result: Distribution = {};
    const allKeys = new Set<string>();
    
    events.forEach(event => {
      Object.keys(event.distribution).forEach(key => allKeys.add(key));
    });

    allKeys.forEach(key => {
      const sum = events.reduce((acc, event) => {
        return acc + (event.distribution[key] || 0);
      }, 0);
      result[key] = sum / events.length;
    });

    return normalizeDistribution(result);
  };

  const computeStableLetter = useCallback((events: ModelEvent[]): LetterEvent | null => {
    if (events.length === 0) return null;

    const avgDist = averageDistributions(events);
    const entries = Object.entries(avgDist).sort((a, b) => b[1] - a[1]);
    
    if (entries.length === 0) return null;

    const [letter, confidence] = entries[0];
    const alternatives = entries.slice(1, 4).map(([l]) => l);

    return {
      letter,
      confidence,
      distribution: avgDist,
      alternatives,
    };
  }, []);

  const pushModelEvent = useCallback((event: ModelEvent) => {
    if (isLockedRef.current) return;

    setBuffer(prev => {
      const normalized = {
        ...event,
        distribution: normalizeDistribution(event.distribution),
      };

      const newBuffer = [...prev, normalized];
      if (newBuffer.length > bufferSize) {
        newBuffer.shift();
      }

      const stable = computeStableLetter(newBuffer);
      if (stable && stable.confidence >= emissionThreshold) {
        setCurrentLetter(stable);
      }

      return newBuffer;
    });
  }, [bufferSize, emissionThreshold, computeStableLetter]);

  const lockEmission = useCallback((locked: boolean) => {
    isLockedRef.current = locked;
  }, []);

  const acceptLetter = useCallback((letter: LetterEvent) => {
    setHistory(prev => {
      const newHistory = prev.slice(0, historyIndex + 1);
      return [...newHistory, letter];
    });
    setHistoryIndex(prev => prev + 1);
    setCurrentLetter(null);
    setBuffer([]);
  }, [historyIndex]);

  const getPrev = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex(prev => prev - 1);
      return history[historyIndex - 1];
    }
    return null;
  }, [history, historyIndex]);

  const getNext = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(prev => prev + 1);
      return history[historyIndex + 1];
    }
    return null;
  }, [history, historyIndex]);

  const reset = useCallback(() => {
    setBuffer([]);
    setCurrentLetter(null);
    setHistory([]);
    setHistoryIndex(-1);
    isLockedRef.current = false;
  }, []);

  return {
    currentLetter,
    history: history.slice(0, historyIndex + 1),
    pushModelEvent,
    lockEmission,
    acceptLetter,
    getPrev,
    getNext,
    reset,
  };
}
