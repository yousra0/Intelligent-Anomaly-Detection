"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/AuthContext";
import { Navbar } from "@/components/layout/Navbar";
import { Logo } from "@/components/ui/Logo";
import { Loader2 } from "lucide-react";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.replace("/login");
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Logo size="lg" showWordmark={false} />
          <Loader2 className="h-5 w-5 animate-spin text-pwc-orange" />
        </div>
      </div>
    );
  }

  if (!isAuthenticated) return null;

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      {/*
        Desktop-first main area:
        - w-full fills entire viewport width
        - max-w-screen-2xl caps at 1536px on very wide screens
        - px scales up with breakpoints for comfortable reading
        - No forced margins that would require browser zoom
      */}
      <main className="mx-auto w-full max-w-screen-2xl flex-1 px-4 py-6 sm:px-6 md:px-8 lg:px-10 xl:px-12">
        {children}
      </main>
    </div>
  );
}
