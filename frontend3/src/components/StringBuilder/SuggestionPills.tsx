import React from "react";

type Props = {
  suggestions: string[];
  onAcceptWord: (word: string) => void;
};

export const SuggestionPills: React.FC<Props> = ({ suggestions, onAcceptWord }) => {
  if (!suggestions || suggestions.length === 0) return <div style={{ marginTop: 8 }}>No suggestions</div>;
  return (
    <div className="suggestion-row" style={{ marginTop: 8 }}>
      {suggestions.map((word, idx) => (
        <button key={idx} className="suggestion-pill" onClick={() => onAcceptWord(word)} style={{ marginRight: 6 }}>
          {word}
        </button>
      ))}
    </div>
  );
};
