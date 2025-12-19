import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Navigation from "@/components/Navigation";
import { Hand } from "lucide-react";

const Library = () => {
  const alphabets = Array.from("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
  const numbers = Array.from("0123456789");
  const commonWords = ["Hello", "Thank You", "Please", "Yes", "No", "Help", "Sorry", "Good", "Bad", "More"];

  const GestureCard = ({ label, category }: { label: string; category: string }) => (
    <Card className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1 overflow-hidden border-border/50 cursor-pointer">
      <div className="aspect-square bg-gradient-to-br from-primary/5 to-accent/5 flex items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-accent/10 opacity-0 group-hover:opacity-100 transition-opacity" />
        <Hand className="h-16 w-16 text-muted-foreground/30 group-hover:text-primary/40 transition-colors" />
        <div className="absolute bottom-2 right-2">
          <Badge variant="secondary" className="text-xs">
            {category}
          </Badge>
        </div>
      </div>
      <div className="p-4 bg-card">
        <h3 className="font-semibold text-center text-lg">{label}</h3>
        <p className="text-xs text-muted-foreground text-center mt-1">
          Tap to view details
        </p>
      </div>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/30">
      <Navigation />
      
      <div className="container px-4 py-12 max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Gesture{" "}
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Reference Library
            </span>
          </h1>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Browse the complete collection of Indian Sign Language gestures. 
            Learn alphabets, numbers, and common words.
          </p>
        </div>

        <Tabs defaultValue="alphabets" className="space-y-8">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-3 h-12">
            <TabsTrigger value="alphabets" className="text-base">Alphabets</TabsTrigger>
            <TabsTrigger value="numbers" className="text-base">Numbers</TabsTrigger>
            <TabsTrigger value="words" className="text-base">Words</TabsTrigger>
          </TabsList>

          <TabsContent value="alphabets" className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-muted-foreground">
                {alphabets.length} gestures available
              </p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {alphabets.map((letter) => (
                <GestureCard key={letter} label={letter} category="Alphabet" />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="numbers" className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-muted-foreground">
                {numbers.length} gestures available
              </p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {numbers.map((num) => (
                <GestureCard key={num} label={num} category="Number" />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="words" className="space-y-4">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-muted-foreground">
                {commonWords.length} common words
              </p>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {commonWords.map((word) => (
                <GestureCard key={word} label={word} category="Word" />
              ))}
            </div>
          </TabsContent>
        </Tabs>

        <Card className="mt-12 p-8 bg-gradient-to-br from-primary/5 to-accent/5 border-primary/10">
          <div className="text-center space-y-4">
            <h3 className="text-2xl font-bold">Need More Gestures?</h3>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our library is continuously growing. Images and videos for each gesture 
              will be integrated with the backend to provide detailed learning materials.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default Library;
