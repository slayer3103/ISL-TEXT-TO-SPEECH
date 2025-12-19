import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { LetterEvent } from "@/hooks/useLetterStreamAdapter";

interface LetterConfirmProps {
  letterEvent: LetterEvent | null;
}

export const LetterConfirm = ({ letterEvent }: LetterConfirmProps) => {
  if (!letterEvent) {
    return (
      <Card className="p-6 bg-muted/30">
        <div className="text-center text-muted-foreground">
          <p className="text-sm">Waiting for prediction...</p>
        </div>
      </Card>
    );
  }

  const confidencePercent = Math.round(letterEvent.confidence * 100);

  return (
    <Card className="p-6 border-2 shadow-soft">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-lg">Pending Letter</h3>
          <Badge 
            variant={confidencePercent >= 70 ? "default" : "secondary"}
            className="text-sm"
          >
            {confidencePercent}% confidence
          </Badge>
        </div>

        <div className="p-8 bg-gradient-to-br from-primary/10 to-accent/10 rounded-lg border border-primary/20 text-center">
          <div className="text-6xl font-bold text-foreground mb-2">
            {letterEvent.letter}
          </div>
          {letterEvent.alternatives.length > 0 && (
            <div className="flex gap-2 justify-center mt-4">
              {letterEvent.alternatives.map((alt, idx) => (
                <Badge key={idx} variant="outline" className="text-lg px-3 py-1">
                  {alt}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
