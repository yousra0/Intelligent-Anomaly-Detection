import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { QueryProvider } from "@/providers/QueryProvider";
import { AuthProvider } from "@/lib/auth/AuthContext";
import { Toaster } from "sonner";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "PwC Audit Analytics Platform",
  description: "Plateforme interne de détection d'anomalies financières — PwC Tunisie",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <QueryProvider>
          <AuthProvider>
            {children}
            <Toaster
              position="top-right"
              richColors
              toastOptions={{ duration: 4000 }}
            />
          </AuthProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
