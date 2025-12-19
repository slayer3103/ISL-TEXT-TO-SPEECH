import { Badge } from "@/components/ui/badge";
import { Lightbulb } from "lucide-react";

interface SuggestionPillsProps {
  suggestions: string[];
  onSelect: (word: string) => void;
}

export const SuggestionPills = ({ suggestions, onSelect }: SuggestionPillsProps) => {
  if (suggestions.length === 0) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Lightbulb className="h-4 w-4" />
        <span>Suggestions</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((word, idx) => (
          <Badge
            key={idx}
            variant="secondary"
            className="cursor-pointer hover:bg-primary hover:text-primary-foreground transition-colors px-3 py-1.5 text-sm"
            onClick={() => onSelect(word)}
          >
            {word}
          </Badge>
        ))}
      </div>
    </div>
  );
};
