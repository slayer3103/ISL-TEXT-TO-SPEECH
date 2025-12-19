// src/components/SuggestionPills.tsx
import React from "react";

type Props = {
  suggestions: string[];
  onAcceptWord: (word: string) => void;
};

export function SuggestionPills({ suggestions, onAcceptWord }: Props) {
  if (!suggestions || suggestions.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
      {suggestions.map((s, i) => (
        <button
          key={i}
          onClick={() => onAcceptWord(s)}
          style={{
            padding: "8px 12px",
            borderRadius: 999,
            background: "#eef2ff",
            border: "1px solid #ddd",
            cursor: "pointer",
            fontWeight: 600
          }}
        >
          {s}
        </button>
      ))}
    </div>
  );
}

export default SuggestionPills;
