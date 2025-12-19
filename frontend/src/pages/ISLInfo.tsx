import { Card } from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import { Users, Globe, BookOpen, Lightbulb } from "lucide-react";

const ISLInfo = () => {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <Navigation />
      
      <div className="container px-4 py-12 max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            About{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Indian Sign Language
            </span>
          </h1>
          <p className="text-muted-foreground text-lg">
            Understanding the visual language that connects millions
          </p>
        </div>

        <div className="space-y-8">
          <Card className="p-8 shadow-soft">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary to-primary-glow flex items-center justify-center shrink-0">
                <Users className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-3">What is ISL?</h2>
                <p className="text-muted-foreground leading-relaxed">
                  Indian Sign Language (ISL) is the primary sign language used by the deaf community 
                  in India. It's a complete, natural language with its own grammar and syntax, 
                  distinct from spoken languages like Hindi or English. ISL uses hand shapes, 
                  movements, facial expressions, and body language to convey meaning.
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-8 shadow-soft">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-accent to-accent-glow flex items-center justify-center shrink-0">
                <Globe className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-3">ISL in India</h2>
                <div className="text-muted-foreground space-y-3">
                  <p className="leading-relaxed">
                    With over 5 million deaf individuals in India, ISL serves as a vital communication 
                    bridge. In 2021, ISL was officially recognized in the National Education Policy, 
                    marking a significant step toward inclusivity and accessibility.
                  </p>
                  <div className="grid md:grid-cols-2 gap-4 mt-4">
                    <div className="p-4 bg-primary/5 rounded-lg border border-primary/10">
                      <div className="text-2xl font-bold text-primary mb-1">5M+</div>
                      <div className="text-sm text-muted-foreground">Deaf individuals in India</div>
                    </div>
                    <div className="p-4 bg-accent/5 rounded-lg border border-accent/10">
                      <div className="text-2xl font-bold text-accent mb-1">2021</div>
                      <div className="text-sm text-muted-foreground">Official recognition year</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-8 shadow-soft">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center shrink-0">
                <BookOpen className="h-6 w-6 text-white" />
              </div>
              <div className="w-full">
                <h2 className="text-2xl font-bold mb-4">Key Characteristics</h2>
                <div className="grid md:grid-cols-2 gap-4">
                  {[
                    {
                      title: "Regional Variations",
                      description: "Like spoken languages, ISL has regional dialects across different states"
                    },
                    {
                      title: "Visual Grammar",
                      description: "Uses space, movement, and facial expressions to convey grammatical structure"
                    },
                    {
                      title: "Manual & Non-manual",
                      description: "Combines hand signs with facial expressions and body movements"
                    },
                    {
                      title: "Independent Language",
                      description: "Not based on Hindi or English; has its own unique linguistic structure"
                    }
                  ].map((item, idx) => (
                    <div key={idx} className="p-4 bg-muted/50 rounded-lg">
                      <h3 className="font-semibold mb-2">{item.title}</h3>
                      <p className="text-sm text-muted-foreground">{item.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          <Card className="p-8 shadow-soft">
            <div className="flex items-start gap-4 mb-4">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-accent to-primary flex items-center justify-center shrink-0">
                <Lightbulb className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold mb-3">Recognition Challenges</h2>
                <p className="text-muted-foreground leading-relaxed mb-4">
                  Automated ISL recognition faces unique technical challenges that our AI system addresses:
                </p>
                <ul className="space-y-2 text-muted-foreground">
                  <li className="flex gap-2">
                    <span className="text-primary shrink-0">•</span>
                    <span><strong className="text-foreground">Complex hand shapes:</strong> Subtle differences in finger positions can change meaning entirely</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary shrink-0">•</span>
                    <span><strong className="text-foreground">Movement dynamics:</strong> Speed, direction, and path of movement are crucial</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary shrink-0">•</span>
                    <span><strong className="text-foreground">Two-handed coordination:</strong> Many signs require precise synchronization of both hands</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-primary shrink-0">•</span>
                    <span><strong className="text-foreground">Lighting & background:</strong> Environmental factors affect visual recognition accuracy</span>
                  </li>
                </ul>
              </div>
            </div>
          </Card>

          <Card className="p-8 bg-gradient-to-r from-primary to-accent text-white shadow-accent">
            <h2 className="text-2xl font-bold mb-3">Our Mission</h2>
            <p className="leading-relaxed opacity-90">
              We believe technology should break down communication barriers. By developing AI-powered 
              ISL recognition, we're working to make communication more accessible and inclusive for 
              the deaf community in India. This project represents a step toward a world where everyone 
              can communicate freely, regardless of hearing ability.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ISLInfo;
