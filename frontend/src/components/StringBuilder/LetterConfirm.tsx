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
  if (!letterEvent) return <div className="pending-letter">Waiting for stable letter…</div>;

  const showChar = letterEvent.char === " " || letterEvent.char === "" ? "<blank>" : letterEvent.char;
  const percent = Math.round(letterEvent.confidence * 10000) / 100;

  return (
    <div className="pending-letter">
      <div className="letter-pill">
        {showChar} <span className="conf">({percent}% )</span>
      </div>
      <div className="actions" style={{ marginTop: 8 }}>
        <button onClick={onAccept}>Accept</button>
        <button onClick={onReject}>Reject</button>
      </div>
      <div className="alternatives" style={{ marginTop: 6 }}>
        {letterEvent.alternatives.map((alt, idx) => (
          <span key={idx} className="alt-pill" style={{ marginRight: 6 }}>{alt.char}({Math.round(alt.confidence * 10000) / 100}%)</span>
        ))}
      </div>
    </div>
  );
};
