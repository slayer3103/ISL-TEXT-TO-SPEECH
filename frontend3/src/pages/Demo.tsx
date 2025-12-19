// frontend/src/pages/Demo.tsx
import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Navigation from "@/components/Navigation";
import { Volume2, VolumeX, Video, VideoOff, Activity } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
const SEND_EVERY_MS = 200; // pipeline interval
const SMOOTH_WINDOW = 6;
const VOTE_THRESHOLD = 4;
const CONF_SPEAK_THRESH = 60;
const MIN_SPEECH_INTERVAL = 1.2; // seconds

// Sequence client-side feature params
const SEQ_T = 48;
const SEQ_RUN_EVERY = 3;
const POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24];

// Single-frame frequency (send every N processed results)
const SINGLE_SEND_EVERY = 3;

// commit cooldown (secs) to avoid repeated identical commits stacking
const COMMIT_COOLDOWN = 1.0;

const Demo: React.FC = () => {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    if (!loading && !user) navigate("/login");
  }, [user, loading, navigate]);

  // UI state
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(isMuted);
  useEffect(() => { isMutedRef.current = isMuted; }, [isMuted]);

  const [cameraActive, setCameraActive] = useState(false);
  const [prediction, setPrediction] = useState("Ready to recognize...");
  const [confidence, setConfidence] = useState(0);

  // new UI features
  const [autoCommit, setAutoCommit] = useState(false);
  const [seqInFlight, setSeqInFlight] = useState(false);
  const [singleInFlight, setSingleInFlight] = useState(false);

  // Video + MP
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null); // hidden for single-frame capture
  const intervalRef = useRef<number | null>(null);
  const holRef = useRef<any | null>(null);

  // buffers and features
  const sequenceMode = useRef<boolean>(false);
  const featsBufRef = useRef<number[][]>([]);
  const maskBufRef = useRef<number[]>([]);
  const frameCounterRef = useRef<number>(0);

  // smoothing + commit control
  const labelBufRef = useRef<string[]>([]);
  const lastCommittedRef = useRef<{ label: string; ts: number }>({ label: "", ts: 0 });

  // request concurrency & stale response avoidance
  const seqRequestIdRef = useRef<number>(0); // increments for each seq request
  const singleRequestIdRef = useRef<number>(0);

  // prev coords for velocity
  const prevLeftRef = useRef<number[] | null>(null);
  const prevRightRef = useRef<number[] | null>(null);
  const prevTsRef = useRef<number | null>(null);

  // String builder state
  const [builderTokens, setBuilderTokens] = useState<string[]>([]);
  const builderHistoryRef = useRef<string[][]>([]);
  const builderSelectionRef = useRef<number>(-1); // for prev/next (not used heavily now)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (holRef.current && typeof holRef.current.close === "function") {
        try { holRef.current.close(); } catch {}
        holRef.current = null;
      }
    };
  }, []);

  // Dynamic import for MediaPipe Holistic
  async function initHolistic() {
    if (holRef.current) return holRef.current;
    try {
      const mp = await import("@mediapipe/holistic");
      const HolisticCtor = (mp as any).Holistic || (mp as any).default?.Holistic || (mp as any);
      const hol = new HolisticCtor({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
      });
      hol.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.4,
        minTrackingConfidence: 0.4,
        refineFaceLandmarks: false,
      });
      hol.onResults((res: any) => {
        try { onHolisticResults(res); } catch (e) { console.error("onHolisticResults error", e); }
      });
      holRef.current = hol;
      console.info("Holistic initialized");
      return hol;
    } catch (e) {
      console.error("initHolistic failed:", e);
      holRef.current = null;
      return null;
    }
  }

  // Start camera
  async function startCamera() {
    console.info("startCamera: requesting getUserMedia...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "user" },
        audio: false,
      });
      if (!videoRef.current) {
        console.error("startCamera: videoRef unexpectedly null");
        return;
      }
      videoRef.current.srcObject = stream;
      try { await videoRef.current.play(); } catch {}
      if (!holRef.current) await initHolistic();

      // ensure canvas exists for single capture
      if (!canvasRef.current) {
        const c = document.createElement("canvas");
        canvasRef.current = c;
      }

      featsBufRef.current = []; maskBufRef.current = []; frameCounterRef.current = 0;
      prevLeftRef.current = null; prevRightRef.current = null; prevTsRef.current = null;

      if (intervalRef.current) window.clearInterval(intervalRef.current);
      intervalRef.current = window.setInterval(captureAndProcessFrame, SEND_EVERY_MS) as unknown as number;

      setCameraActive(true);
      console.info("Camera started and interval set");

      // quick backend health ping
      try {
        fetch(`${API_URL}/health`).then(r=>r.json()).then(js=>console.info("Backend health:", js)).catch(e=>console.warn("Backend health fetch failed", e));
      } catch (e) { console.warn("health check failed", e); }
    } catch (err) {
      console.error("startCamera error:", err);
      setCameraActive(false);
    }
  }

  function stopCamera() {
    console.info("stopCamera called");
    if (intervalRef.current) { window.clearInterval(intervalRef.current); intervalRef.current = null; }
    if (videoRef.current && videoRef.current.srcObject) {
      const s = videoRef.current.srcObject as MediaStream;
      s.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }

  // Helpers to flatten landmarks
  function lmToFlat(landmarks: any): number[] {
    if (!landmarks || landmarks.length === 0) return new Array(42).fill(0);
    const out: number[] = [];
    for (let i = 0; i < Math.min(21, landmarks.length); i++) {
      out.push(landmarks[i].x ?? 0);
      out.push(landmarks[i].y ?? 0);
    }
    while (out.length < 42) out.push(0);
    return out;
  }
  function poseToFlat(poseLandmarks: any): number[] {
    if (!poseLandmarks) return new Array(POSE_IDX.length * 2).fill(0);
    const out: number[] = [];
    for (const idx of POSE_IDX) {
      const lm = poseLandmarks[idx];
      out.push(lm ? (lm.x ?? 0) : 0);
      out.push(lm ? (lm.y ?? 0) : 0);
    }
    return out;
  }

  // Periodically send video into holistic
  async function captureAndProcessFrame() {
    try {
      const video = videoRef.current;
      const hol = holRef.current;
      if (!video) return;
      if (video.readyState < 2) return;
      if (!hol) return;
      try {
        await hol.send({ image: video });
      } catch (e) {
        console.warn("hol.send failed:", e);
      }
    } catch (err) {
      console.error("captureAndProcessFrame error:", err);
    }
  }

  // Called by holistic when results are ready
  function onHolisticResults(results: any) {
    try {
      const leftFlat = lmToFlat(results.leftHandLandmarks ?? []);
      const rightFlat = lmToFlat(results.rightHandLandmarks ?? []);
      const poseFlat = poseToFlat(results.poseLandmarks ?? []);
      const leftPresent = (results.leftHandLandmarks && results.leftHandLandmarks.length > 0) ? 1 : 0;
      const rightPresent = (results.rightHandLandmarks && results.rightHandLandmarks.length > 0) ? 1 : 0;

      const nowSec = performance.now() / 1000;
      let lw_vel = [0, 0];
      let rw_vel = [0, 0];
      if (prevTsRef.current != null) {
        const dt = Math.max(1e-3, nowSec - prevTsRef.current!);
        if (prevLeftRef.current) lw_vel = [(leftFlat[0] - prevLeftRef.current[0]) / dt, (leftFlat[1] - prevLeftRef.current[1]) / dt];
        if (prevRightRef.current) rw_vel = [(rightFlat[0] - prevRightRef.current[0]) / dt, (rightFlat[1] - prevRightRef.current[1]) / dt];
      }
      prevLeftRef.current = leftFlat.slice();
      prevRightRef.current = rightFlat.slice();
      prevTsRef.current = nowSec;

      let dist = 0;
      if (leftFlat[0] !== 0 && rightFlat[0] !== 0) {
        dist = Math.hypot(leftFlat[0] - rightFlat[0], leftFlat[1] - rightFlat[1]);
      }

      const feat = [
        ...leftFlat, ...rightFlat, ...poseFlat,
        lw_vel[0], lw_vel[1], rw_vel[0], rw_vel[1],
        dist, leftPresent, rightPresent,
      ];

      // push to client-side buffers
      featsBufRef.current.push(feat);
      maskBufRef.current.push((leftPresent + rightPresent) > 0 ? 1 : 0);
      if (featsBufRef.current.length > SEQ_T) { featsBufRef.current.shift(); maskBufRef.current.shift(); }
      frameCounterRef.current = (frameCounterRef.current || 0) + 1;

      // Sequence mode: send features to backend (with concurrency & stale-check)
      if (sequenceMode.current) {
        if ((frameCounterRef.current % SEQ_RUN_EVERY) === 0 && featsBufRef.current.length >= Math.min(SEQ_T, featsBufRef.current.length)) {
          // increment id and mark inflight
          seqRequestIdRef.current += 1;
          const reqId = seqRequestIdRef.current;
          setSeqInFlight(true);
          const payload = { feats_arr: featsBufRef.current.slice(), mask: maskBufRef.current.slice() };
          console.debug(`[seq send] id=${reqId} len=${payload.feats_arr.length}`);
          fetch(`${API_URL}/infer_seq_feats`, {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
          })
            .then((r) => { if (!r.ok) throw new Error(`status ${r.status}`); return r.json(); })
            .then((js) => {
              // discard stale responses
              if (reqId !== seqRequestIdRef.current) {
                console.debug(`[seq discard] id=${reqId} current=${seqRequestIdRef.current}`);
                setSeqInFlight(false);
                return;
              }
              setSeqInFlight(false);
              if (!js) return;
              const label = js.label ?? js.name ?? "";
              const conf = Number(js.confidence ?? js.conf ?? 0);
              if (label) { setPrediction(String(label)); setConfidence(Math.round(conf)); }
              // smoothing + commit (NO AUTO TTS)
              if (label) {
                const buf = labelBufRef.current; buf.push(String(label)); if (buf.length > SMOOTH_WINDOW) buf.shift();
                const counts: Record<string, number> = {}; for (const v of buf) counts[v] = (counts[v] || 0) + 1;
                const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
                const [topLabel, topCount] = entries[0] ?? ["", 0];
                if (topCount >= VOTE_THRESHOLD && conf >= CONF_SPEAK_THRESH) {
                  const now = performance.now() / 1000;
                  const last = lastCommittedRef.current;
                  if (autoCommit) {
                    // commit to builder (with cooldown)
                    if (last.label !== topLabel || now - last.ts > COMMIT_COOLDOWN) {
                      lastCommittedRef.current = { label: topLabel, ts: now };
                      pushBuilderToken(topLabel);
                    }
                  } else {
                    // update lastCommittedRef for cooldown purposes (but do not append)
                    lastCommittedRef.current = { label: topLabel, ts: now };
                  }
                }
              }
            })
            .catch((e) => {
              setSeqInFlight(false);
              console.warn("/infer_seq_feats fetch failed", e);
            });
        }
      } else {
        // SINGLE mode: send image (to /infer) every SINGLE_SEND_EVERY processed frames
        if ((frameCounterRef.current % SINGLE_SEND_EVERY) === 0) {
          singleRequestIdRef.current += 1;
          const reqId = singleRequestIdRef.current;
          setSingleInFlight(true);
          // capture a JPEG from hidden canvas and POST to /infer
          try {
            const v = videoRef.current;
            const c = canvasRef.current || document.createElement("canvas");
            c.width = v?.videoWidth || 640;
            c.height = v?.videoHeight || 480;
            const ctx = c.getContext("2d");
            if (ctx && v) {
              ctx.save();
              // mirror to match displayed video
              ctx.translate(c.width, 0); ctx.scale(-1, 1);
              ctx.drawImage(v, 0, 0, c.width, c.height);
              ctx.restore();
              const dataUrl = c.toDataURL("image/jpeg", 0.7);
              fetch(`${API_URL}/infer`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: dataUrl }),
              })
                .then((r) => { if (!r.ok) throw new Error(`status ${r.status}`); return r.json(); })
                .then((js) => {
                  if (reqId !== singleRequestIdRef.current) {
                    console.debug(`[single discard] id=${reqId} current=${singleRequestIdRef.current}`);
                    setSingleInFlight(false);
                    return;
                  }
                  setSingleInFlight(false);
                  if (!js) return;
                  const label = js.label ?? js.name ?? "";
                  const conf = Number(js.confidence ?? js.conf ?? 0);
                  if (label) { setPrediction(String(label)); setConfidence(Math.round(conf)); }
                  // NOTE: NO auto-speak nor auto-commit on single mode
                })
                .catch((e) => {
                  setSingleInFlight(false);
                  console.warn("/infer fetch failed", e);
                });
            } else {
              setSingleInFlight(false);
            }
          } catch (e) {
            setSingleInFlight(false);
            console.warn("single capture failed", e);
          }
        }
      }
    } catch (e) { console.error("onHolisticResults exception:", e); }
  }

  // Backend mode notify (server may flush queues)
  async function notifyBackendMode(mode: "single" | "sequence") {
    try {
      const r = await fetch(`${API_URL}/set_mode`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode }),
      });
      if (!r.ok) console.warn("/set_mode failed", r.status);
    } catch (e) { console.warn("/set_mode fetch error", e); }
  }

  // String builder control helpers
  function pushBuilderToken(tok: string) {
    builderHistoryRef.current.push([...builderTokens]);
    setBuilderTokens((prev) => [...prev, tok]);
  }
  function addSpace() { pushBuilderToken(" "); }
  function backspace() {
    builderHistoryRef.current.push([...builderTokens]);
    setBuilderTokens((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.length <= 1) return prev.slice(0, -1);
      const newPrev = prev.slice(0, -1);
      newPrev.push(last.slice(0, last.length - 1));
      return newPrev;
    });
  }
  function undo() {
    const hist = builderHistoryRef.current;
    if (hist.length === 0) return;
    const last = hist.pop();
    if (last) setBuilderTokens(last);
  }
  function clearBuilder() {
    builderHistoryRef.current.push([...builderTokens]);
    setBuilderTokens([]);
  }
  function speakBuilder() {
    const text = builderTokens.join("");
    if (!text) return;
    if ("speechSynthesis" in window) {
      try {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(u);
      } catch (e) { console.warn("TTS failed", e); }
    }
  }

  // Accept current prediction into builder (no TTS)
  function acceptPrediction() {
    if (!prediction || prediction === "Ready to recognize..." || prediction === "<UNKNOWN>") return;
    pushBuilderToken(prediction);
  }

  // prev/next navigation in builder history (simple)
  function prevBuilder() {
    const idx = Math.max(0, builderSelectionRef.current - 1);
    builderSelectionRef.current = idx;
    const hist = builderHistoryRef.current;
    if (hist.length > idx) setBuilderTokens(hist[idx]);
  }
  function nextBuilder() {
    const idx = Math.min(builderHistoryRef.current.length - 1, builderSelectionRef.current + 1);
    builderSelectionRef.current = idx;
    const hist = builderHistoryRef.current;
    if (hist.length > idx) setBuilderTokens(hist[idx]);
  }

  // toggle handlers
  const handleToggleCamera = () => {
    if (cameraActive) stopCamera();
    else startCamera();
  };
  const handleToggleMute = () => setIsMuted((s) => !s);
  const handleToggleMode = async () => {
    sequenceMode.current = !sequenceMode.current;
    setPrediction(sequenceMode.current ? "Sequence mode" : "Single-frame mode");
    await notifyBackendMode(sequenceMode.current ? "sequence" : "single");
  };

  // Render
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="container mx-auto px-4 py-6">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-semibold">Live ISL Translator</h2>
            <p className="text-sm text-muted-foreground">Translate hand gestures in real-time</p>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="secondary">{sequenceMode.current ? "Sequence" : "Single"}</Badge>
            <div className="text-right">
              <div className="text-lg font-medium">{prediction}</div>
              <div className="text-xs text-muted-foreground">Confidence: {confidence}%</div>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Video + controls */}
          <div className="lg:col-span-2 space-y-4">
            <Card className="overflow-hidden border-2 shadow-soft">
              <div className="aspect-video bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center relative">
                <div className="w-full h-full flex items-center justify-center relative">
                  <video
                    ref={videoRef}
                    className="w-full h-full object-cover"
                    playsInline
                    autoPlay
                    muted
                    style={{ transform: "scaleX(-1)" }}
                  />
                  {!cameraActive && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/30 text-white pointer-events-none">
                      <VideoOff className="h-16 w-16 opacity-80" />
                      <p className="mt-3">Camera is off</p>
                      <p className="text-sm opacity-70">Click "Start Camera" to begin</p>
                    </div>
                  )}
                </div>
              </div>
            </Card>

            <div className="flex gap-3 flex-wrap items-center">
              <Button size="lg" onClick={handleToggleCamera} className={cameraActive ? "bg-destructive hover:bg-destructive/90 shadow-glow" : ""}>
                {cameraActive ? (<><VideoOff className="h-5 w-5 mr-2" /> Stop Camera</>) : (<><Video className="h-5 w-5 mr-2" /> Start Camera</>)}
              </Button>

              <Button size="lg" variant="outline" onClick={handleToggleMute}>
                {isMuted ? (<><VolumeX className="h-5 w-5 mr-2" />Unmute</>) : (<><Volume2 className="h-5 w-5 mr-2" />Mute</>)}
              </Button>

              <Button size="lg" variant="ghost" onClick={handleToggleMode}>
                {sequenceMode.current ? (<><Activity className="h-5 w-5 mr-2" /> Sequence</>) : (<><Video className="h-5 w-5 mr-2" /> Single</>)}
              </Button>

              {/* In-flight badges */}
              <div className="ml-3 flex items-center gap-2">
                <div className="text-xs">Seq:</div>
                <Badge variant={seqInFlight ? "destructive" : "secondary"}>{seqInFlight ? "In-flight" : "Idle"}</Badge>
                <div className="text-xs ml-2">Single:</div>
                <Badge variant={singleInFlight ? "destructive" : "secondary"}>{singleInFlight ? "In-flight" : "Idle"}</Badge>
              </div>

              {/* Auto-commit toggle */}
              <div className="ml-3 flex items-center gap-2">
                <label className="text-sm text-muted-foreground">Auto-commit</label>
                <input type="checkbox" checked={autoCommit} onChange={(e)=>setAutoCommit(e.target.checked)} />
              </div>
            </div>

            <Card className="mt-4 p-4">
              <div className="text-sm text-muted-foreground">Tips</div>
              <ul className="list-disc ml-4 text-sm mt-2 text-muted-foreground/90">
                <li>Open browser console (F12) to see logs and errors.</li>
                <li>Make sure <code>VITE_API_URL</code> points to your backend (default: http://localhost:5000).</li>
                <li>If MediaPipe fails to initialize, run <code>npm install @mediapipe/holistic</code> or allow CDN load.</li>
                <li>Sequence mode uses client-side features and calls <code>/infer_seq_feats</code>. Single mode sends frames to <code>/infer</code>.</li>
              </ul>
            </Card>
          </div>

          {/* Right column: builder + status */}
          <div className="space-y-4">
            <Card className="p-4">
              <div className="text-sm text-muted-foreground mb-2">String Builder</div>
              <div className="min-h-[4rem] p-2 bg-white/5 rounded">
                <div className="break-words">{builderTokens.length ? builderTokens.join("") : <span className="text-muted-foreground">(empty)</span>}</div>
              </div>

              <div className="flex flex-wrap gap-2 mt-3">
                <Button size="sm" onClick={acceptPrediction} disabled={!prediction || prediction === "<UNKNOWN>"}>Accept</Button>
                <Button size="sm" variant="outline" onClick={() => pushBuilderToken(" ")}>Space</Button>
                <Button size="sm" variant="outline" onClick={backspace}>Backspace</Button>
                <Button size="sm" variant="ghost" onClick={undo}>Undo</Button>
                <Button size="sm" onClick={() => { if (prediction) { pushBuilderToken(prediction); setPrediction("Ready to recognize..."); }}}>Prev</Button>
                <Button size="sm" onClick={() => { /* next placeholder */ }}>Next</Button>
                <Button size="sm" variant="destructive" onClick={clearBuilder}>Clear</Button>
                <Button size="sm" onClick={speakBuilder}>Speak</Button>
              </div>
            </Card>

            <Card className="p-4">
              <div className="text-sm text-muted-foreground">Status</div>
              <div className="mt-2">
                <div className="text-xs text-muted-foreground">Mode</div>
                <div className="font-medium">{sequenceMode.current ? "Sequence (word)" : "Single (letter/alpha)"}</div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="text-sm text-muted-foreground">Confidence</div>
              <div className="mt-2 text-lg font-semibold">{confidence}%</div>
            </Card>
          </div>
        </div>
      </div>

      {/* Hidden canvas for single-frame capture */}
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
};

export default Demo;
