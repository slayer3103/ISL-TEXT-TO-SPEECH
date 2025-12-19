// File: src/components/StringBuilder/SuggestionPills.tsx

import React from "react";

type Props = {
  suggestions: string[];
  onAcceptWord: (word: string) => void;
};

export const SuggestionPills: React.FC<Props> = ({ suggestions, onAcceptWord }) => {
  return (
    <div className="suggestion-row">
      {suggestions.map((word, idx) => (
        <button key={idx} className="suggestion-pill" onClick={() => onAcceptWord(word)}>
          {word}
        </button>
      ))}
    </div>
  );
};
