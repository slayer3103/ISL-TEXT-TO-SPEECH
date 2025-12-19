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
const SEND_EVERY_MS = 300; // send a frame every 300ms (reduce load)
const SMOOTH_WINDOW = 6; // smoothing window size (number of recent labels)
const VOTE_THRESHOLD = 4; // number of votes in window required to commit / speak
const CONF_SPEAK_THRESH = 60; // minimum confidence percent to speak
const MIN_SPEECH_INTERVAL = 1.2; // seconds between repeated TTS of same label

// Sequence settings (added)
const SEQ_T = 48; // number of frames to send for sequence inference
const SEQ_RUN_EVERY = 3; // send sequence every N captured frames

const Demo: React.FC = () => {
  const [isMuted, setIsMuted] = useState(false);
  const isMutedRef = useRef(isMuted); // keep latest mute state for loops
  const [cameraActive, setCameraActive] = useState(false);
  const [prediction, setPrediction] = useState("Ready to recognize...");
  const [confidence, setConfidence] = useState(0);
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !user) {
      navigate("/login");
    }
  }, [user, loading, navigate]);

  // Keep isMutedRef in sync with state
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  // Sequence-related refs
  const sequenceMode = useRef<boolean>(false); // toggle sequence vs single-frame
  const frameBufRef = useRef<string[]>([]);
  const frameCounterRef = useRef<number>(0);

  // smoothing buffer and last spoken
  const labelBufRef = useRef<string[]>([]);
  const lastCommittedRef = useRef<{ label: string; ts: number }>({ label: "", ts: 0 });

  useEffect(() => {
    if (cameraActive) startCamera();
    else stopCamera();
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraActive]);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 960, height: 540, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => {});
      // ensure canvas exists
      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
      }
      // start periodic capture - reset buffer & counter for sequence mode
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
      frameBufRef.current = [];
      frameCounterRef.current = 0;
      intervalRef.current = window.setInterval(captureAndSendFrame, SEND_EVERY_MS) as unknown as number;
    } catch (err) {
      console.error("Camera start error:", err);
      setCameraActive(false);
    }
  }

  function stopCamera() {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      } catch (e) {}
    }
  }

  async function captureAndSendFrame() {
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      if (video.readyState < 2) return;

      // Mirror both display and sent frame:
      // video element is mirrored via CSS; canvas must draw mirrored image so backend receives same orientation.
      canvas.width = video.videoWidth || 960;
      canvas.height = video.videoHeight || 540;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // draw mirrored: translate + scale
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

      // push to local buffer for possible sequence inference
      frameBufRef.current.push(dataUrl);
      if (frameBufRef.current.length > SEQ_T) frameBufRef.current.shift();
      frameCounterRef.current = (frameCounterRef.current || 0) + 1;

      // If sequence mode is enabled, send sequence every SEQ_RUN_EVERY captures
      let jsResp: any = null;
      if (sequenceMode.current) {
        if ((frameCounterRef.current % SEQ_RUN_EVERY) === 0 && frameBufRef.current.length >= Math.min(SEQ_T, frameBufRef.current.length)) {
          // send a copy (oldest->newest)
          const framesToSend = frameBufRef.current.slice();
          try {
            const resp = await fetch(`${API_URL}/infer_seq`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ frames: framesToSend }),
            });
            if (!resp.ok) return;
            const js = await resp.json();
            if (!js) return;
            jsResp = js;
          } catch (e) {
            console.warn("Sequence infer failed", e);
            return;
          }
        } else {
          // not time to send sequence yet; skip single-frame inference
          return;
        }
      } else {
        // single-frame mode: send only the latest frame
        try {
          const resp = await fetch(`${API_URL}/infer`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: dataUrl }),
          });
          if (!resp.ok) return;
          const js = await resp.json();
          if (!js) return;
          jsResp = js;
        } catch (e) {
          console.warn("Infer failed", e);
          return;
        }
      }

      if (!jsResp) return;

      const label = jsResp.label ?? jsResp.name ?? "";
      const conf = Number(jsResp.confidence ?? jsResp.conf ?? 0);

      // Update UI quickly
      if (label) {
        setPrediction(String(label));
        setConfidence(Math.round(conf));
      }

      // Smoothing: maintain circular buffer of last SMOOTH_WINDOW labels
      if (label) {
        const buf = labelBufRef.current;
        buf.push(String(label));
        if (buf.length > SMOOTH_WINDOW) buf.shift();

        // count occurrences
        const counts: Record<string, number> = {};
        for (const v of buf) counts[v] = (counts[v] || 0) + 1;
        const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const [topLabel, topCount] = entries[0] || ["", 0];

        // Commit + TTS only when the topLabel has enough votes and high confidence
        if (topCount >= VOTE_THRESHOLD && conf >= CONF_SPEAK_THRESH) {
          const now = performance.now() / 1000;
          if (lastCommittedRef.current.label !== topLabel || now - lastCommittedRef.current.ts > MIN_SPEECH_INTERVAL) {
            lastCommittedRef.current = { label: topLabel, ts: now };
            // speak if not muted (use ref to avoid stale closure)
            if (!isMutedRef.current && "speechSynthesis" in window) {
              try {
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance(topLabel);
                window.speechSynthesis.speak(u);
              } catch (e) {
                console.warn("TTS failed", e);
              }
            }
          }
        }
      }
    } catch (err) {
      // capture/send errors are non-fatal - keep running
      // console.error("capture/send error", err);
    }
  }

  const handleToggleCamera = () => setCameraActive((s) => !s);
  const handleToggleMute = () => setIsMuted((s) => !s);

  // UI rendering
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
          {/* Video Preview */}
          <div className="lg:col-span-2 space-y-4">
            <Card className="overflow-hidden border-2 shadow-soft">
              <div className="aspect-video bg-gradient-to-br from-muted to-muted/50 flex items-center justify-center relative">
                {!cameraActive ? (
                  <div className="text-center space-y-4 p-8">
                    <VideoOff className="h-16 w-16 text-muted-foreground mx-auto opacity-50" />
                    <div>
                      <p className="text-muted-foreground text-lg">Camera is off</p>
                      <p className="text-sm text-muted-foreground/70">Click "Start Camera" below to begin</p>
                    </div>
                  </div>
                ) : (
                  <div className="w-full h-full bg-gradient-to-b from-primary/10 to-accent/10 flex items-center justify-center">
                    {/* Mirror visually with CSS transform; capture code also mirrors before sending */}
                    <video
                      ref={videoRef}
                      className="w-full h-full object-cover"
                      playsInline
                      autoPlay
                      muted
                      style={{ transform: "scaleX(-1)" }}
                    />
                  </div>
                )}
              </div>
            </Card>

            {/* Controls */}
            <div className="flex gap-3 flex-wrap">
              <Button
                size="lg"
                onClick={handleToggleCamera}
                className={cameraActive ? "bg-destructive hover:bg-destructive/90 shadow-glow" : ""}
              >
                {cameraActive ? (
                  <>
                    <VideoOff className="h-5 w-5 mr-2" />
                    Stop Camera
                  </>
                ) : (
                  <>
                    <Video className="h-5 w-5 mr-2" />
                    Start Camera
                  </>
                )}
              </Button>

              <Button size="lg" variant="outline" onClick={handleToggleMute} disabled={!cameraActive} className="hover:bg-secondary/80">
                {isMuted ? (
                  <>
                    <VolumeX className="h-5 w-5 mr-2" />
                    Unmute
                  </>
                ) : (
                  <>
                    <Volume2 className="h-5 w-5 mr-2" />
                    Mute
                  </>
                )}
              </Button>

              <Button
                size="lg"
                variant="ghost"
                onClick={() => {
                  sequenceMode.current = !sequenceMode.current;
                  setPrediction(sequenceMode.current ? "Sequence mode" : "Single-frame mode");
                }}
                disabled={!cameraActive}
                className="hover:bg-secondary/80"
              >
                {sequenceMode.current ? (
                  <>
                    <Activity className="h-5 w-5 mr-2" />
                    Sequence
                  </>
                ) : (
                  <>
                    <Video className="h-5 w-5 mr-2" />
                    Single
                  </>
                )}
              </Button>
            </div>

            <Card className="mt-4 p-4">
              <div className="text-sm text-muted-foreground">Tips</div>
              <ul className="list-disc ml-4 text-sm mt-2 text-muted-foreground/90">
                <li>Sequence mode collects frames and sends a clip to the backend (`/infer_seq`) for word LSTM inference.</li>
                <li>Single mode sends a single frame to `/infer` (letters / alphabet / landmark MLP).</li>
                <li>Sequence mode is heavier on server (many frames get processed by MediaPipe) — consider `/infer_seq_feats` if you want to compute features client-side.</li>
              </ul>
            </Card>
          </div>

          {/* Right column: info */}
          <div className="space-y-4">
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

            <Card className="p-4">
              <div className="text-sm text-muted-foreground">Controls</div>
              <div className="mt-2">
                <div className="text-xs text-muted-foreground">Interval</div>
                <div className="text-sm">{SEND_EVERY_MS} ms per capture</div>
                <div className="text-xs text-muted-foreground mt-2">Sequence frames</div>
                <div className="text-sm">{SEQ_T} frames / send every {SEQ_RUN_EVERY} captures</div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Demo;
