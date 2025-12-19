import { Button } from "@/components/ui/button";
import { Undo, Delete, Volume2, RotateCcw, Play } from "lucide-react";
import { Card } from "@/components/ui/card";

interface ControlsProps {
  onUndo: () => void;
  onBackspace: () => void;
  onSpace: () => void;
  onFinalize: () => void;
  onReset: () => void;
  canUndo: boolean;
  text: string;
}

export const Controls = ({
  onUndo,
  onBackspace,
  onSpace,
  onFinalize,
  onReset,
  canUndo,
  text,
}: ControlsProps) => {
  return (
    <Card className="p-6 bg-muted/30">
      <div className="space-y-3">
        <h3 className="font-semibold mb-3">Controls</h3>
        
        <div className="grid grid-cols-2 gap-2">
          <Button
            onClick={onSpace}
            variant="secondary"
            size="lg"
            className="w-full"
          >
            Space
          </Button>
          
          <Button
            onClick={onBackspace}
            variant="secondary"
            size="lg"
            className="w-full"
            disabled={text.length === 0}
          >
            <Delete className="h-5 w-5 mr-2" />
            Backspace
          </Button>
          
          <Button
            onClick={onUndo}
            variant="secondary"
            size="lg"
            className="w-full"
            disabled={!canUndo}
          >
            <Undo className="h-5 w-5 mr-2" />
            Undo
          </Button>
          
          <Button
            onClick={onReset}
            variant="secondary"
            size="lg"
            className="w-full"
          >
            <RotateCcw className="h-5 w-5 mr-2" />
            Reset
          </Button>
        </div>

        <Button
          onClick={onFinalize}
          disabled={text.trim().length === 0}
          className="w-full bg-gradient-to-r from-accent to-primary hover:opacity-90"
          size="lg"
        >
          <Volume2 className="h-5 w-5 mr-2" />
          Speak Text
        </Button>
      </div>
    </Card>
  );
};
