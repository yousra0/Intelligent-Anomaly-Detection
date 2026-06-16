"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useAuth } from "@/lib/auth/AuthContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Lock, Mail, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

const schema = z.object({
  email: z
    .string()
    .email("Adresse email invalide.")
    .endsWith("@pwc.com", "Seules les adresses @pwc.com sont autorisées."),
  password: z.string().min(1, "Mot de passe requis."),
});

type FormValues = z.infer<typeof schema>;

export default function LoginPage() {
  const router = useRouter();
  const params = useSearchParams();
  const { login, isAuthenticated, isLoading: authLoading } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      router.replace(params.get("from") ?? "/missions");
    }
  }, [isAuthenticated, authLoading, router, params]);

  const onSubmit = async (values: FormValues) => {
    setServerError(null);
    try {
      await login(values);
      router.replace(params.get("from") ?? "/missions");
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { error?: string } } })?.response?.data?.error ??
        "Identifiants invalides.";
      setServerError(msg);
    }
  };

  if (authLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-pwc-orange" />
      </div>
    );
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-[#F7F7F7] px-4">
      {/* PwC brand header */}
      <div className="mb-8 flex flex-col items-center gap-3">
        <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-pwc-orange shadow-lg">
          <span className="text-xl font-black text-white">PwC</span>
        </div>
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900">Audit Analytics Platform</h1>
          <p className="mt-1 text-sm text-gray-500">Détection d'anomalies financières — Usage interne</p>
        </div>
      </div>

      {/* Login card */}
      <Card className="w-full max-w-sm shadow-lg">
        <CardHeader className="space-y-1 pb-4">
          <CardTitle className="text-xl">Connexion</CardTitle>
          <CardDescription>
            Identifiez-vous avec votre compte PwC
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {/* Server error */}
            {serverError && (
              <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {serverError}
              </div>
            )}

            {/* Email */}
            <div className="space-y-1.5">
              <Label htmlFor="email">Adresse email</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="prenom.nom@pwc.com"
                  autoComplete="username"
                  className={cn("pl-9", errors.email && "border-destructive")}
                  {...register("email")}
                />
              </div>
              {errors.email && (
                <p className="text-xs text-destructive">{errors.email.message}</p>
              )}
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <Label htmlFor="password">Mot de passe</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  autoComplete="current-password"
                  className={cn("pl-9", errors.password && "border-destructive")}
                  {...register("password")}
                />
              </div>
              {errors.password && (
                <p className="text-xs text-destructive">{errors.password.message}</p>
              )}
            </div>

            <Button
              type="submit"
              className="w-full"
              size="lg"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : null}
              Se connecter
            </Button>
          </form>

        </CardContent>
      </Card>

      <p className="mt-6 text-center text-xs text-gray-400">
        © {new Date().getFullYear()} PricewaterhouseCoopers. Usage interne uniquement.
      </p>
    </main>
  );
}
