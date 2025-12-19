// src/components/demo/Controls.tsx
import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ChevronLeft, ChevronRight, X, RotateCw, ArrowLeftCircle, Volume2, Trash2 } from "lucide-react";

type Props = {
  onAccept?: () => void;
  onPrev?: () => void;
  onNext?: () => void;
  onUndo?: () => void;
  onBackspace?: () => void;
  onSpace?: () => void;
  onFinalize?: () => void;
  onReset?: () => void;

  // flags
  canAccept?: boolean;
  hasPrev?: boolean;
  hasNext?: boolean;
  canUndo?: boolean;
  text?: string;
  locked?: boolean;
};

export const Controls: React.FC<Props> = ({
  onAccept,
  onPrev,
  onNext,
  onUndo,
  onBackspace,
  onSpace,
  onFinalize,
  onReset,
  canAccept = false,
  hasPrev = false,
  hasNext = false,
  canUndo = false,
  text = "",
  locked = false,
}) => {
  return (
    <Card className="p-4">
      <div className="space-y-3">
        {/* Accept / Primary */}
        <div>
          <Button
            onClick={() => !locked && onAccept?.()}
            className={`w-full ${canAccept && !locked ? "bg-primary text-white" : "bg-muted-foreground/10 text-muted-foreground"}`}
            disabled={!canAccept || locked}
          >
            ✓ Accept Letter
          </Button>
        </div>

        {/* Prev / Next */}
        <div className="grid grid-cols-2 gap-2 mt-2">
          <Button onClick={() => onPrev?.()} disabled={!hasPrev} size="sm" className="justify-center">
            <ChevronLeft className="mr-2" /> Prev
          </Button>
          <Button onClick={() => onNext?.()} disabled={!hasNext} size="sm" className="justify-center">
            Next <ChevronRight className="ml-2" />
          </Button>
        </div>

        {/* Space / Backspace */}
        <div className="grid grid-cols-2 gap-2 mt-2">
          <Button onClick={() => onSpace?.()} size="sm" className="justify-center">
            Space
          </Button>
          <Button onClick={() => onBackspace?.()} size="sm" className="justify-center" disabled={!text || text.length === 0}>
            ⌫ Backspace
          </Button>
        </div>

        {/* Undo / Reset */}
        <div className="grid grid-cols-2 gap-2 mt-2">
          <Button onClick={() => onUndo?.()} disabled={!canUndo} size="sm" className="justify-center">
            ↶ Undo
          </Button>
          <Button onClick={() => onReset?.()} size="sm" className="justify-center">
            ⟳ Reset
          </Button>
        </div>

        {/* Finalize / Speak */}
        <div className="mt-3">
          <Button onClick={() => onFinalize?.()} className="w-full bg-gradient-to-r from-orange-400 to-violet-400 text-white" disabled={!text || text.trim().length === 0}>
            <Volume2 className="mr-2" /> Speak Text
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default Controls;
