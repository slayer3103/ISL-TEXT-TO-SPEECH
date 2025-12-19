import { Card } from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import { Brain, Code, Zap, Target } from "lucide-react";

const About = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <Navigation />
      
      <div className="container px-4 py-12 max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            About{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Our Project
            </span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Building bridges through AI-powered sign language recognition
          </p>
        </div>

        <div className="space-y-8">
          <Card className="p-8 shadow-soft">
            <h2 className="text-2xl font-bold mb-4">The Vision</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Communication is a fundamental human right. Our project aims to leverage cutting-edge 
              artificial intelligence to create real-time Indian Sign Language recognition that makes 
              communication accessible to everyone. By combining computer vision, deep learning, and 
              natural language processing, we're building technology that understands and translates 
              sign language instantly.
            </p>
            <p className="text-muted-foreground leading-relaxed">
              This platform serves as both an educational tool and a practical communication aid, 
              helping to bridge the gap between the deaf and hearing communities while promoting 
              awareness and understanding of ISL.
            </p>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">
            <Card className="p-6 shadow-soft hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary to-primary-glow flex items-center justify-center mb-4">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">AI Models</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Our system uses advanced deep learning models trained on thousands of ISL gestures. 
                We employ MediaPipe for hand tracking and custom neural networks for gesture classification, 
                achieving high accuracy even in varying lighting conditions.
              </p>
            </Card>

            <Card className="p-6 shadow-soft hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-accent to-accent-glow flex items-center justify-center mb-4">
                <Code className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">Technology Stack</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Built with Python (PyTorch, MediaPipe) for the ML backend and React for the frontend. 
                We use TorchScript for efficient model inference and offline text-to-speech for accessibility. 
                The system processes video frames in real-time with minimal latency.
              </p>
            </Card>

            <Card className="p-6 shadow-soft hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center mb-4">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">Real-time Processing</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Our pipeline processes webcam frames every 200ms, extracting hand landmarks and running 
                inference on both alphabet and word models. Temporal smoothing reduces prediction flicker, 
                providing a stable and reliable recognition experience.
              </p>
            </Card>

            <Card className="p-6 shadow-soft hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-accent to-primary flex items-center justify-center mb-4">
                <Target className="h-6 w-6 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">Future Goals</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                We're continuously improving accuracy, expanding vocabulary, and adding support for 
                continuous sign language sentences. Future versions will include mobile apps, 
                multi-user support, and integration with video conferencing platforms.
              </p>
            </Card>
          </div>

          <Card className="p-8 shadow-soft">
            <h2 className="text-2xl font-bold mb-4">How It Works</h2>
            <div className="space-y-6">
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-bold text-sm">
                  1
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Video Capture & Preprocessing</h3>
                  <p className="text-muted-foreground text-sm">
                    Webcam frames are captured and preprocessed to normalize lighting and resolution. 
                    The system detects hands using MediaPipe Holistic for robust tracking.
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-bold text-sm">
                  2
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Feature Extraction</h3>
                  <p className="text-muted-foreground text-sm">
                    Hand landmarks (21 points per hand) are extracted and normalized. These coordinates 
                    form feature vectors that capture hand shape, orientation, and position.
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-bold text-sm">
                  3
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Model Inference</h3>
                  <p className="text-muted-foreground text-sm">
                    Features are passed through TorchScript models (alphabet and word classifiers). 
                    The system applies temporal smoothing to reduce jitter and improve accuracy.
                  </p>
                </div>
              </div>

              <div className="flex gap-4">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-bold text-sm">
                  4
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Output & Speech</h3>
                  <p className="text-muted-foreground text-sm">
                    Predicted text is displayed instantly. Offline TTS generates audio output, 
                    allowing users to hear the recognized signs spoken aloud.
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-8 bg-gradient-to-br from-muted/50 to-muted/30 border-border/50">
            <h2 className="text-2xl font-bold mb-3">Limitations & Future Work</h2>
            <p className="text-muted-foreground mb-4 leading-relaxed">
              While our system performs well with individual signs, continuous sentence recognition 
              remains challenging. We're actively working on:
            </p>
            <ul className="grid md:grid-cols-2 gap-3 text-sm text-muted-foreground">
              <li className="flex gap-2">
                <span className="text-primary">•</span>
                <span>Expanding vocabulary beyond alphabets and basic words</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary">•</span>
                <span>Improving accuracy in low-light conditions</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary">•</span>
                <span>Adding support for regional ISL variations</span>
              </li>
              <li className="flex gap-2">
                <span className="text-primary">•</span>
                <span>Implementing sentence-level recognition with grammar</span>
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default About;
