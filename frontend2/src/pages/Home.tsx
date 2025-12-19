import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Video, BookOpen, Zap, Shield } from "lucide-react";
import Navigation from "@/components/Navigation";

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <Navigation />
      
      {/* Hero Section */}
      <section className="container px-4 py-20 md:py-32">
        <div className="mx-auto max-w-4xl text-center space-y-8">
          <div className="inline-block px-4 py-2 bg-primary/10 rounded-full text-sm font-medium text-primary mb-4 animate-fade-in">
            Breaking Communication Barriers
          </div>
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight">
            Bridge the Gap with{" "}
            <span className="bg-gradient-to-r from-primary via-primary-glow to-accent bg-clip-text text-transparent">
              Sign Language AI
            </span>
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto">
            Real-time Indian Sign Language recognition powered by advanced AI. 
            Translate gestures to text and speech instantly, making communication accessible for everyone.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Link to="/demo">
              <Button size="lg" className="bg-gradient-to-r from-primary to-primary-glow hover:opacity-90 transition-opacity shadow-glow text-base px-8">
                Try Live Demo
              </Button>
            </Link>
            <Link to="/library">
              <Button size="lg" variant="outline" className="text-base px-8 hover:bg-secondary/80 transition-colors">
                Browse Gestures
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container px-4 py-20">
        <div className="mx-auto max-w-6xl">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
            Powerful Features for Seamless Communication
          </h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Video,
                title: "Live Recognition",
                description: "Real-time gesture recognition using your webcam with high accuracy",
                gradient: "from-primary to-primary-glow"
              },
              {
                icon: Zap,
                title: "Instant Translation",
                description: "Fast prediction with text-to-speech output for immediate communication",
                gradient: "from-accent to-accent-glow"
              },
              {
                icon: BookOpen,
                title: "Gesture Library",
                description: "Comprehensive reference of ISL alphabets, numbers, and common words",
                gradient: "from-primary to-accent"
              },
              {
                icon: Shield,
                title: "Privacy First",
                description: "Your video stays on your device. No data is stored or shared",
                gradient: "from-accent to-primary"
              }
            ].map((feature, idx) => (
              <Card key={idx} className="p-6 hover:shadow-lg transition-all duration-300 hover:-translate-y-1 border-border/50 bg-card/50 backdrop-blur">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 shadow-soft`}>
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="font-semibold text-lg mb-2">{feature.title}</h3>
                <p className="text-muted-foreground text-sm">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="container px-4 py-20 bg-gradient-subtle">
        <div className="mx-auto max-w-4xl text-center space-y-12">
          <div>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-muted-foreground text-lg">
              Our AI-powered system recognizes Indian Sign Language in three simple steps
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8 text-left">
            {[
              { step: "01", title: "Enable Camera", description: "Grant camera access and position yourself in frame" },
              { step: "02", title: "Make Gestures", description: "Perform ISL signs - alphabets, numbers, or words" },
              { step: "03", title: "Get Results", description: "See predictions instantly with optional voice output" }
            ].map((item, idx) => (
              <div key={idx} className="space-y-3">
                <div className="text-5xl font-bold bg-gradient-to-br from-primary to-primary-glow bg-clip-text text-transparent opacity-60">
                  {item.step}
                </div>
                <h3 className="font-semibold text-xl">{item.title}</h3>
                <p className="text-muted-foreground">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="container px-4 py-20">
        <div className="mx-auto max-w-3xl text-center space-y-6 bg-gradient-to-r from-primary to-accent p-12 rounded-2xl shadow-accent text-white">
          <h2 className="text-3xl md:text-4xl font-bold">Ready to Get Started?</h2>
          <p className="text-lg opacity-90">
            Experience the power of AI-driven sign language recognition. No installation required.
          </p>
          <Link to="/demo">
            <Button size="lg" className="bg-white text-primary hover:bg-white/90 transition-colors text-base px-8 shadow-lg">
              Launch Demo Now
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-muted/30 py-8">
        <div className="container px-4 text-center text-sm text-muted-foreground">
          <p>© 2025 SignSpeak. Making communication accessible through AI.</p>
        </div>
      </footer>
    </div>
  );
};

export default Home;
