import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Navigation from "@/components/Navigation";
import { Video, VideoOff, Activity } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { useLetterStreamAdapter } from "@/hooks/useLetterStreamAdapter";
import { useInterval } from "@/hooks/useInterval";
import { useUndoStack } from "@/hooks/useUndoStack";
import { SuggestionEngine } from "@/lib/suggestionEngine";
import { LetterConfirm } from "@/components/demo/LetterConfirm";
import { StringBuilder } from "@/components/demo/StringBuilder";
import { SuggestionPills } from "@/components/demo/SuggestionPills";
import { Controls } from "@/components/demo/Controls";
import { toast } from "sonner";
import wordlistData from "@/data/wordlist.json";
const INFER_ENDPOINT = import.meta.env.VITE_INFER_ENDPOINT || "http://localhost:5000/infer";
const CAPTURE_INTERVAL = 200; // ms

const Demo = () => {
  const [cameraActive, setCameraActive] = useState(false);
  const [text, setText] = useState("");
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sbControlRef = useRef<any>(null);

  const suggestionEngineRef = useRef<SuggestionEngine>(new SuggestionEngine(wordlistData as string[]));
  const {
    currentLetter,
    history,
    pushModelEvent,
    lockEmission,
    acceptLetter,
    getPrev,
    getNext,
    reset: resetAdapter
  } = useLetterStreamAdapter({
    bufferSize: 5,
    emissionThreshold: 0.6
  });

  const {
    current: textStack,
    canUndo,
    push: pushText,
    undo: undoText,
    reset: resetStack
  } = useUndoStack("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isLocked, setIsLocked] = useState(false);
  const [historyIndex, setHistoryIndex] = useState(0);

  useEffect(() => {
    if (!loading && !user) {
      navigate("/login");
    }
  }, [user, loading, navigate]);

  // Update text from history and reset navigation index
  useEffect(() => {
    const newText = history.map((e) => e.letter).join("");
    setText(newText);
    setHistoryIndex(0); // Reset to current when history changes
  }, [history]);

  // Update suggestions based on current word
  useEffect(() => {
    const words = text.split(" ");
    const currentWord = words[words.length - 1];
    if (currentWord.length >= 2) {
      const newSuggestions = suggestionEngineRef.current.getSuggestions(currentWord, currentLetter?.distribution);
      setSuggestions(newSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [text, currentLetter]);

  // --- Option A: set cameraActive early, then attach stream when videoRef exists ---
  const startCamera = async () => {
    try {
      // request camera stream first
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false
      });

      // store stream and make UI show the video element so videoRef can mount
      streamRef.current = stream;
      setCameraActive(true);

      // wait for the video element to be present (short polling, up to 2s)
      const waitForVideoRef = async () => {
        const start = Date.now();
        while (!videoRef.current) {
          await new Promise((r) => setTimeout(r, 30));
          if (Date.now() - start > 2000) break;
        }
      };
      await waitForVideoRef();

      if (!videoRef.current) {
        // video element didn't mount in time
        console.warn("startCamera: videoRef still missing after render");
        toast.error("Could not attach camera preview (video element not ready).");
        return;
      }

      // attach stream and satisfy autoplay policy
      videoRef.current.srcObject = stream;
      try { videoRef.current.muted = true; } catch (e) { /* ignore */ }
      try { (videoRef.current as any).playsInline = true; } catch (e) { /* ignore */ }

      // attempt to play
      try {
        const p = videoRef.current.play();
        if (p && typeof (p as any).then === "function") {
          await (p as Promise<void>).catch((e) => {
            console.warn("video.play promise rejected:", e);
          });
        }
      } catch (e) {
        console.warn("video.play threw:", e);
      }

      setCameraActive(true); // reaffirm
      console.info("Camera started, video size:", videoRef.current?.videoWidth, "x", videoRef.current?.videoHeight);
      toast.success("Camera started");
    } catch (err: any) {
      console.error("Camera error:", err);
      toast.error("Failed to access camera - " + (err?.message || err));
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      try {
        videoRef.current.pause();
      } catch (e) {}
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    toast.info("Camera stopped");
  };

  // --- captureAndInfer with validation & logging ---
  const captureAndInfer = async () => {
    try {
      if (!videoRef.current || !canvasRef.current || !cameraActive) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // ensure video ready
      if (video.videoWidth === 0 || video.videoHeight === 0 || video.readyState < 2) {
        // not ready yet
        return;
      }

      // ensure canvas matches video size
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataURL = canvas.toDataURL("image/jpeg", 0.8);

      if (!dataURL || dataURL.length < 200) {
        console.warn("Captured image is too small or empty", dataURL ? dataURL.length : 0);
        return;
      }

      // debug log
      console.debug("Sending frame to infer endpoint, sizeKB:", Math.round(dataURL.length / 1024));

      try {
        const resp = await fetch(INFER_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ frame: dataURL, speak: false })
        });

        if (!resp.ok) {
          console.warn("Inference endpoint returned non-ok:", resp.status, resp.statusText);
          try {
            const text = await resp.text();
            console.debug("Infer non-ok body:", text);
          } catch (e) {}
          return;
        }

        const data = await resp.json();
        console.debug("Infer response:", data);

        const alternatives = (data.alternatives || []).map((alt: any) => (typeof alt === "string" ? alt : alt.label || alt));

        pushModelEvent({
          distribution: data.distribution || {},
          label: data.label || "",
          confidence: data.confidence || 0,
          alternatives
        });
      } catch (err) {
        console.error("Error sending frame to infer:", err);
      }
    } catch (err) {
      console.error("captureAndInfer unexpected error:", err);
    }
  };

  useInterval(cameraActive ? captureAndInfer : null, CAPTURE_INTERVAL);

  const handleAccept = () => {
    if (!currentLetter) return;
    setIsLocked(true);
    lockEmission(true);
    acceptLetter(currentLetter);
    pushText(text + currentLetter.letter);
    setTimeout(() => {
      setIsLocked(false);
      lockEmission(false);
    }, 300);
  };

  const handleReject = () => {
    setHistoryIndex(0); // Reset to current
  };

  const handlePrev = () => {
    if (historyIndex > 0) {
      const prevLetter = getPrev();
      if (prevLetter) {
        setHistoryIndex(historyIndex - 1);
      }
    }
  };

  const handleNext = () => {
    const maxIndex = 4; // Buffer size from adapter config
    if (historyIndex < maxIndex) {
      const nextLetter = getNext();
      if (nextLetter) {
        setHistoryIndex(historyIndex + 1);
      }
    }
  };

  const handleSpace = () => {
    pushText(text + " ");
    setText(text + " ");
  };

  const handleBackspace = () => {
    if (text.length > 0) {
      const newText = text.slice(0, -1);
      pushText(newText);
      setText(newText);
    }
  };

  const handleUndo = () => {
    undoText();
    setText(textStack);
  };

  const handleReset = () => {
    resetAdapter();
    resetStack("");
    setText("");
    setSuggestions([]);
    toast.info("Reset complete");
  };

  const handleFinalize = () => {
    if (text.trim().length === 0) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
    toast.success("Speaking text...");
  };

  const handleSuggestionSelect = (word: string) => {
    const words = text.split(" ");
    words[words.length - 1] = word;
    const newText = words.join(" ") + " ";
    pushText(newText);
    setText(newText);
    toast.success(`Selected: ${word}`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted/30 flex items-center justify-center">
        <div className="text-center">
          <Activity className="h-12 w-12 text-primary mx-auto animate-pulse mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <Navigation />

      <div className="container px-4 py-12 max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            ISL Real-Time{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Interpreter</span>
          </h1>
        </div>

        <div className="grid lg:grid-cols-[1fr_400px] gap-6">
          {/* Left Column: Video Preview & Text Builder */}
          <div className="space-y-4">
            <Card className="overflow-hidden border-2 shadow-soft">
  {/* Larger fixed-height preview to make video easier to see */}
  <div className="bg-gradient-to-br from-muted to-muted/50 relative w-full h-[520px] md:h-[480px] lg:h-[520px]">
    {cameraActive ? (
      <>
        {/* video fills the container */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover scale-x-[-1]"
        />
        <canvas ref={canvasRef} className="hidden" />
      </>
    ) : (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center space-y-4 p-8">
          <VideoOff className="h-16 w-16 text-muted-foreground mx-auto opacity-50" />
          <div>
            <p className="text-muted-foreground font-medium mb-2">Camera is off</p>
            <p className="text-sm text-muted-foreground/70">Select mode and start camera</p>
          </div>
        </div>
      </div>
    )}

    {cameraActive && (
      <div className="absolute top-4 left-4 flex gap-2">
        <Badge className="bg-accent/90 text-white shadow-sm">
          <span className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse" />
          Live
        </Badge>
      </div>
    )}
  </div>
</Card>


            {/* Camera Controls */}
<div className="flex gap-3 flex-wrap">
  <Button
    size="lg"
    onClick={() => (cameraActive ? stopCamera() : startCamera())}
    className={cameraActive ? "bg-destructive hover:bg-destructive/90" : "bg-gradient-to-r from-primary to-primary-glow hover:opacity-90 shadow-glow"}
  >
    {cameraActive ? (
      <>
        <VideoOff className="h-5 w-5 mr-2" />
        Stop
      </>
    ) : (
      <>
        <Video className="h-5 w-5 mr-2" />
        Start Camera
      </>
    )}
  </Button>

  {/* --- Debug buttons (temporary) --- */}
  <div className="flex items-center gap-2">
    <Button size="sm" onClick={() => { console.log("dbg: prev"); handlePrev(); }}>Prev</Button>
    <Button size="sm" onClick={() => { console.log("dbg: next"); handleNext(); }}>Next</Button>
    <Button size="sm" onClick={() => { console.log("dbg: space"); handleSpace(); }}>Space</Button>
    <Button size="sm" onClick={() => { console.log("dbg: backspace"); handleBackspace(); }}>Backspace</Button>
    <Button size="sm" onClick={() => { console.log("dbg: undo"); handleUndo(); }}>Undo</Button>
  </div>
</div>


            {/* String Builder */}
            <StringBuilder
              value={text}
              onChange={setText}
              letterEvent={currentLetter}
              suggestionEngine={suggestionEngineRef.current}
              controlRef={sbControlRef}
            />

            {/* Suggestions */}
            {suggestions.length > 0 && (
              <Card className="p-4">
                <SuggestionPills suggestions={suggestions} onSelect={handleSuggestionSelect} />
              </Card>
            )}
          </div>

          {/* Right Column: Prediction & Controls */}
          <div className="space-y-4">
            {/* Current Prediction Card */}
            <LetterConfirm letterEvent={currentLetter} />

            {/* Action Controls */}
            <Controls
              onAccept={handleAccept}
              onPrev={handlePrev}
              onNext={handleNext}
              onUndo={handleUndo}
              onBackspace={handleBackspace}
              onSpace={handleSpace}
              onFinalize={handleFinalize}
              onReset={handleReset}
              canAccept={!!currentLetter}
              hasPrev={historyIndex > 0}
              hasNext={historyIndex < 4 && history.length > historyIndex + 1}
              canUndo={canUndo}
              text={text}
              locked={isLocked}
            />

            {/* Usage Guide */}
            <Card className="p-5 bg-muted/30 border-primary/10">
              <h3 className="font-semibold mb-3 text-sm">Button Guide</h3>
              <div className="space-y-2 text-xs text-muted-foreground">
                <div className="flex gap-2">
                  <span className="text-primary font-bold">Accept:</span>
                  <span>Confirm predicted letter & add to text</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-primary font-bold">Space:</span>
                  <span>Insert space to separate words</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-primary font-bold">Backspace:</span>
                  <span>Remove last character from text</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-primary font-bold">Undo:</span>
                  <span>Revert last acceptance action</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-primary font-bold">Prev/Next:</span>
                  <span>Navigate prediction history</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-primary font-bold">Speak:</span>
                  <span>Convert text to speech (TTS)</span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};
export default Demo;
