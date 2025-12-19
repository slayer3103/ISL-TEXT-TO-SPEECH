// File: src/components/StringBuilder/LetterConfirm.tsx

import React from "react";

type Props = {
  letterEvent: {
    char: string;
    confidence: number;
    alternatives: Array<{ char: string; confidence: number }>;
    timestamp: number;
  } | null;
  onAccept: () => void;
  onReject: () => void;
};

export const LetterConfirm: React.FC<Props> = ({ letterEvent, onAccept, onReject }) => {
  if (!letterEvent) return <div className="pending-letter">Waiting for letter…</div>;
  return (
    <div className="pending-letter">
      <div className="letter-pill">
        {letterEvent.char} <span className="conf">({letterEvent.confidence.toFixed(2)})</span>
      </div>
      <div className="actions">
        <button onClick={onAccept}>Accept Letter</button>
        <button onClick={onReject}>Reject</button>
      </div>
      <div className="alternatives">
        {letterEvent.alternatives.map((alt, idx) => (
          <span key={idx} className="alt-pill">{alt.char}({alt.confidence.toFixed(2)})</span>
        ))}
      </div>
    </div>
  );
};
