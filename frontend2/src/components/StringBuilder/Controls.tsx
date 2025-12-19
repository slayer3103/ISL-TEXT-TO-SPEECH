// File: src/components/StringBuilder/Controls.tsx

import React from "react";

type Props = {
  onDelete: () => void;
  onUndo: () => void;
  onFinalize: () => void;
};

export const Controls: React.FC<Props> = ({ onDelete, onUndo, onFinalize }) => {
  return (
    <div className="controls">
      <button onClick={onDelete}>Backspace</button>
      <button onClick={onUndo}>Undo</button>
      <button onClick={onFinalize}>Finalize (Speak)</button>
    </div>
  );
};
