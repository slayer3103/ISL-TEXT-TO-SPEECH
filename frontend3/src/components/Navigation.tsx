import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Menu, Hand, LogOut } from "lucide-react";
import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";

const Navigation = () => {
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const { user, signOut } = useAuth();

  const isActive = (path: string) => location.pathname === path;

  const navLinks = [
    { path: "/", label: "Home" },
    { path: "/demo", label: "Live" },
    { path: "/library", label: "Library" },
    { path: "/isl-info", label: "ISL Info" },
    { path: "/about", label: "About" },
  ];

  const NavItems = ({ mobile = false }: { mobile?: boolean }) => (
    <>
      {navLinks.map((link) => (
        <Link
          key={link.path}
          to={link.path}
          onClick={() => mobile && setOpen(false)}
          className={`transition-colors hover:text-primary ${
            isActive(link.path)
              ? "text-primary font-medium"
              : "text-muted-foreground"
          } ${mobile ? "block py-2" : ""}`}
        >
          {link.label}
        </Link>
      ))}
    </>
  );

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-xl">
  <Hand className="h-6 w-6 text-primary" />
  <div className="inline-block">
    <span className="bg-gradient-to-br from-primary to-primary/80 bg-clip-text text-transparent inline-block">
  SignSpeak
</span>
  </div>
</Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex items-center gap-8">
          <div className="flex items-center gap-6">
            <NavItems />
          </div>
          <div className="flex items-center gap-3">
            {user ? (
              <>
                <span className="text-sm text-muted-foreground">
                  {user.email}
                </span>
                <Button variant="ghost" onClick={signOut}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Logout
                </Button>
              </>
            ) : (
              <>
                <Link to="/login">
                  <Button variant="ghost">Login</Button>
                </Link>
                <Link to="/signup">
                  <Button className="bg-gradient-to-r from-primary to-primary-glow hover:opacity-90 transition-opacity shadow-glow">
                    Get Started
                  </Button>
                </Link>
              </>
            )}
          </div>
        </div>

        {/* Mobile Navigation */}
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild className="md:hidden">
            <Button variant="ghost" size="icon">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-64">
            <div className="flex flex-col gap-6 mt-8">
              <NavItems mobile />
              <div className="flex flex-col gap-3 pt-4 border-t">
                {user ? (
                  <>
                    <div className="text-sm text-muted-foreground px-2 truncate">
                      {user.email}
                    </div>
                    <Button variant="ghost" onClick={signOut} className="w-full justify-start">
                      <LogOut className="h-4 w-4 mr-2" />
                      Logout
                    </Button>
                  </>
                ) : (
                  <>
                    <Link to="/login" onClick={() => setOpen(false)}>
                      <Button variant="ghost" className="w-full">
                        Login
                      </Button>
                    </Link>
                    <Link to="/signup" onClick={() => setOpen(false)}>
                      <Button className="w-full bg-gradient-to-r from-primary to-primary-glow">
                        Get Started
                      </Button>
                    </Link>
                  </>
                )}
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </nav>
  );
};

export default Navigation;
