import React from "react";

type Props = {
  onDelete: () => void;
  onUndo: () => void;
  onFinalize: () => void;
};

export const Controls: React.FC<Props> = ({ onDelete, onUndo, onFinalize }) => {
  return (
    <div className="controls" style={{ marginTop: 12 }}>
      <button onClick={onDelete} style={{ marginRight: 6 }}>Backspace</button>
      <button onClick={onUndo} style={{ marginRight: 6 }}>Undo</button>
      <button onClick={onFinalize}>Finalize (Speak)</button>
    </div>
  );
};
