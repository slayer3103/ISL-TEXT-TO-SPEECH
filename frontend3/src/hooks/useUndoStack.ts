import { useState, useCallback } from "react";

export type UndoOp = {
  type: "acceptLetter" | "acceptWord" | "delete";
  payload: { text: string; length: number };
};

export function useUndoStack(initial: string = "") {
  const [undoStack, setUndoStack] = useState<UndoOp[]>([]);

  const push = useCallback((op: UndoOp) => {
    setUndoStack(stack => [...stack, op]);
  }, []);

  const undo = useCallback(() => {
    let op: UndoOp | undefined;
    setUndoStack(stack => {
      const newStack = [...stack];
      op = newStack.pop();
      return newStack;
    });
    return op;
  }, []);

  return { undoStack: undoStack, push, undo };
}
