import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Type } from "lucide-react";

interface StringBuilderProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
}

export const StringBuilder = ({ value, onChange, readOnly = false }: StringBuilderProps) => {
  return (
    <Card className="p-6 border-2 shadow-soft">
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Type className="h-5 w-5 text-primary" />
          <h3 className="font-semibold text-lg">Generated Text</h3>
        </div>
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          readOnly={readOnly}
          className="min-h-[120px] text-lg font-mono resize-none"
          placeholder="Letters will appear here as you sign..."
        />
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>{value.length} characters</span>
          <span>{value.split(/\s+/).filter(Boolean).length} words</span>
        </div>
      </div>
    </Card>
  );
};
