import React, { useMemo, useState, useRef, useEffect } from "react";
import { Trie } from "../lib/trie/trie";
import { SuggestionEngine } from "../lib/suggestions/suggestions";
import { StringBuilder } from "../components/StringBuilder/StringBuilder";
import { useLetterStreamAdapter } from "../hooks/useLetterStream";

export default function Demo() {
  const trie = useMemo(() => new Trie(), []);
  const suggestionEngine = useMemo(() => new SuggestionEngine(trie), [trie]);

  const [cameraOn, setCameraOn] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stringBuilderRef = useRef<any>(null);

  // stable/live/held predictions
  const [stablePrediction, setStablePrediction] = useState<string>("");
  const [stableConfidence, setStableConfidence] = useState<number>(0);
  const [livePrediction, setLivePrediction] = useState<string>("");
  const [liveConfidence, setLiveConfidence] = useState<number>(0);
  const [heldPrediction, setHeldPrediction] = useState<string>("");

  // Generated text (driven by StringBuilder via onCommittedChange)
  const [committedText, setCommittedText] = useState<string>("");

  // Stability settings
  const STABLE_FRAMES = 3;
  const BLANK_FRAMES = 2;
  const MIN_CONF = 0.40;

  // refs for counters
  const prevRawRef = useRef<string | null>(null);
  const sameCountRef = useRef<number>(0);
  const blankCountRef = useRef<number>(0);
  const lastStableRef = useRef<string | null>(null);
  const lastStableConfRef = useRef<number>(0);
  const heldRef = useRef<string | null>(null);

  useEffect(() => { heldRef.current = heldPrediction || null; }, [heldPrediction]);

  const isBlankLabel = (lab: string | null | undefined) => {
    if (!lab) return true;
    const s = String(lab).trim();
    if (s === "") return true;
    const up = s.toUpperCase();
    return up === "<BLANK>" || up === "<UNK>";
  };

  const handleOnLetter = (ev: any) => {
    try {
      const predicted = ev?.predicted ?? ev?.label ?? "";
      const confidence = ev?.confidence ?? ev?.prob ?? 0;
      setLivePrediction(predicted);
      setLiveConfidence(typeof confidence === "number" ? confidence : 0);
    } catch (e) { /* ignore */ }
  };

  const { pushModelEvent, lockEmission, getPrev, getNext, history } = useLetterStreamAdapter({ onLetter: handleOnLetter });

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  async function startCamera() {
    if (cameraOn) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraOn(false);
      return;
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = s;
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        videoRef.current.play();
      }
      setCameraOn(true);
    } catch (e) {
      console.error("camera error", e);
      alert("Unable to access camera. Please allow permission.");
    }
  }

  useEffect(() => {
    if (!canvasRef.current) canvasRef.current = document.createElement("canvas");
  }, []);

  // inference loop (stable + held logic)
  useEffect(() => {
    let id: any = null;
    const API = (import.meta.env.VITE_API_URL || "http://localhost:5000").replace(/\/$/, "");

    async function step() {
      if (!cameraOn || !videoRef.current) return;
      try {
        const video = videoRef.current;
        const w = video.videoWidth || 640;
        const h = video.videoHeight || 480;
        const canvas = canvasRef.current as HTMLCanvasElement;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.save();
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, w, h);
        ctx.restore();

        const data = canvas.toDataURL("image/jpeg", 0.7);
        const resp = await fetch(API + "/infer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame: data })
        });
        if (!resp.ok) return;
        const j = await resp.json();

        const distribution = j.distribution || {};
        const predicted = j.label ?? j.predicted ?? (j.alternatives && j.alternatives[0] && j.alternatives[0].label) ?? "";
        const confidence = typeof j.confidence === "number" ? j.confidence : 0;

        // live display
        setLivePrediction(predicted);
        setLiveConfidence(confidence);

        // stability counters
        const prevRaw = prevRawRef.current;
        if (prevRaw === predicted) sameCountRef.current = (sameCountRef.current || 0) + 1;
        else { prevRawRef.current = predicted; sameCountRef.current = 1; }

        if (isBlankLabel(predicted)) blankCountRef.current = (blankCountRef.current || 0) + 1;
        else blankCountRef.current = 0;

        if (!isBlankLabel(predicted) && sameCountRef.current >= STABLE_FRAMES && confidence >= MIN_CONF) {
          lastStableRef.current = predicted;
          lastStableConfRef.current = confidence;
          setStablePrediction(predicted);
          setStableConfidence(confidence);
        } else {
          if (lastStableRef.current) {
            setStablePrediction(lastStableRef.current);
            setStableConfidence(lastStableConfRef.current || 0);
          } else {
            setStablePrediction("");
            setStableConfidence(0);
          }
        }

        if (blankCountRef.current >= BLANK_FRAMES && lastStableRef.current) {
          if (heldRef.current !== lastStableRef.current) {
            setHeldPrediction(lastStableRef.current);
            heldRef.current = lastStableRef.current;
          }
        }

        const distObj: any = {};
        for (const k in distribution) distObj[k] = distribution[k];
        pushModelEvent && pushModelEvent({ timestamp: Date.now(), distribution: distObj, predicted, confidence });
      } catch (e) {
        // ignore
      }
    }

    if (cameraOn) id = setInterval(step, 350);
    return () => { if (id) clearInterval(id); };
  }, [cameraOn, pushModelEvent]);

  return (
    <div style={{ padding: 24, fontFamily: "Inter, system-ui, sans-serif", color: "#111827" }}>
      <h1 style={{ textAlign: "center", fontSize: 40, marginBottom: 20, fontWeight: 700 }}>
        ISL Real-Time <span style={{ color: "#7c3aed" }}>Interpreter</span>
      </h1>

      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ flex: 3 }}>
          <div style={{ border: "1px solid #e6e6e6", borderRadius: 8, height: 520, display: "flex", alignItems: "center", justifyContent: "center", background: "#fbfbfd", position: "relative" }}>
            <video ref={videoRef} style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 6, transform: cameraOn ? "scaleX(-1)" : "none" }} playsInline muted />
            {!cameraOn && (
              <div style={{ position: "absolute", textAlign: "center", color: "#9aa0a6" }}>
                <div style={{ fontSize: 18, marginBottom: 6 }}>Camera is off</div>
                <div style={{ fontSize: 13 }}>Start camera to begin</div>
              </div>
            )}
          </div>

          <div style={{ marginTop: 12 }}>
            <button onClick={startCamera} style={{ padding: "8px 14px", background: "#4f46e5", color: "white", borderRadius: 8 }}>
              {cameraOn ? "Stop Camera" : "Start Camera"}
            </button>
          </div>

          <div style={{ marginTop: 16 }}>
            <StringBuilder
              ref={stringBuilderRef}
              suggestionEngine={suggestionEngine}
              pushModelEvent={pushModelEvent}
              lockEmission={lockEmission}
              getPrev={getPrev}
              getNext={getNext}
              history={history}
              onCommittedChange={setCommittedText}
            />
          </div>

          <div style={{ marginTop: 16 }}>
            <div style={{ border: "1px solid #e9eee", borderRadius: 8, padding: 12, background: "#fff" }}>
              <h3 style={{ marginTop: 0 }}>Generated Text</h3>
              <div style={{ minHeight: 120, borderRadius: 6, padding: 12, border: "1px solid #f1f1f1", whiteSpace: "pre-wrap", background: "#fbfbfd" }}>
                {committedText || "Letters will appear here as you sign..."}
              </div>
              <div style={{ fontSize: 12, color: "#6b7280", marginTop: 6 }}>
                {(committedText || "").length} characters • {(committedText || "").split(/\s+/).filter(Boolean).length} words
              </div>
            </div>
          </div>
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ border: "1px solid #e9eee", borderRadius: 8, padding: 12, minHeight: 48, textAlign: "center", color: "#6b7280" }}>
            {stablePrediction ? `${stablePrediction} (${(stableConfidence * 100).toFixed(1)}%)` : "Waiting for stable prediction..."}
          </div>

          <div style={{ border: "1px solid #e9eee", borderRadius: 8, padding: 12 }}>
            <h3 style={{ marginTop: 0 }}>Controls</h3>

            {/* Accept button removed */}

            <div style={{ marginBottom: 10, padding: 8, borderRadius: 8, background: "#f7f7fb", border: "1px solid #eef1ff", color: "#374151" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Held preview (captured when hands disband)</div>
              <div style={{ marginTop: 6, fontWeight: 600 }}>{heldPrediction || <span style={{ color: "#9aa0a6" }}>— none yet —</span>}</div>
            </div>

            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <button title="Move caret to previous word" onClick={() => stringBuilderRef.current?.prev()} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#f3f4f6", border: "1px solid #e5e7eb", fontSize: 14 }}>
                ‹ Prev
              </button>
              <button title="Move caret to next word" onClick={() => stringBuilderRef.current?.next()} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#f3f4f6", border: "1px solid #e5e7eb", fontSize: 14 }}>
                Next ›
              </button>
            </div>

            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <button title="Insert held preview at caret" onClick={() => {
                if (!heldPrediction) return alert("No held preview available to insert.");
                stringBuilderRef.current?.insertLastOutput(heldPrediction);
              }} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#e6f6ff", border: "1px solid #cfeefc", color: "#0b63a8", fontWeight: 600 }}>
                ⤓ Insert Preview
              </button>

              <button title="Clear held preview" onClick={() => {
                setHeldPrediction("");
                heldRef.current = null;
              }} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#ffffff", border: "1px solid #e5e7eb", fontSize: 14 }}>
                Clear Preview
              </button>
            </div>

            <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <button title="Insert space between words" onClick={() => stringBuilderRef.current?.space()} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#f3f4f6", border: "1px solid #e5e7eb", fontSize: 14 }}>
                Space
              </button>
              <button title="Remove last character" onClick={() => stringBuilderRef.current?.deleteLast()} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#f3f4f6", border: "1px solid #e5e7eb", fontSize: 14 }}>
                ⌫ Backspace
              </button>
            </div>

            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <button title="Undo last action" onClick={() => stringBuilderRef.current?.undo()} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#f3f4f6", border: "1px solid #e5e7eb", fontSize: 14 }}>
                ↺ Undo
              </button>
              <button title="Reset everything" onClick={() => { if (confirm("Reset text?")) window.location.reload(); }} style={{ flex: 1, padding: 10, borderRadius: 8, background: "#f3f4f6", border: "1px solid #e5e7eb", fontSize: 14 }}>
                ⟲ Reset
              </button>
            </div>

            <div>
              <button title="Convert built text to speech (TTS)" onClick={() => stringBuilderRef.current?.finalize()} style={{ width: "100%", padding: 10, borderRadius: 10, background: "linear-gradient(90deg,#f97316,#8b5cf6)", color: "white", border: "none", fontSize: 15 }}>
                🔊 Speak Text
              </button>
            </div>

            <div style={{ marginTop: 8, fontSize: 12, color: "#6b7280" }}>
              Tip: Use Prev/Next to move the caret between words. Insert Preview will add the held label at the caret.
            </div>
          </div>

          <div style={{ border: "1px solid #e9eee", borderRadius: 8, padding: 12 }}>
            <h3 style={{ marginTop: 0 }}>Button Guide</h3>
            <div style={{ fontSize: 13, color: "#374151" }}>
              <div><b>Prev/Next:</b> Move caret through built sentence (word-level)</div>
              <div><b>Insert:</b> Insert held preview at caret</div>
              <div><b>Space:</b> Insert a space</div>
              <div><b>Backspace:</b> Remove last character</div>
              <div><b>Undo:</b> Revert last action</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
