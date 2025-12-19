import { useState, useCallback } from 'react';

export interface UndoStackState<T> {
  current: T;
  canUndo: boolean;
  canRedo: boolean;
}

export function useUndoStack<T>(initialValue: T) {
  const [history, setHistory] = useState<T[]>([initialValue]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const push = useCallback((value: T) => {
    setHistory(prev => {
      const newHistory = prev.slice(0, currentIndex + 1);
      return [...newHistory, value];
    });
    setCurrentIndex(prev => prev + 1);
  }, [currentIndex]);

  const undo = useCallback(() => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  }, [currentIndex]);

  const redo = useCallback(() => {
    if (currentIndex < history.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  }, [currentIndex, history.length]);

  const reset = useCallback((value: T) => {
    setHistory([value]);
    setCurrentIndex(0);
  }, []);

  return {
    current: history[currentIndex],
    canUndo: currentIndex > 0,
    canRedo: currentIndex < history.length - 1,
    push,
    undo,
    redo,
    reset,
  };
}
