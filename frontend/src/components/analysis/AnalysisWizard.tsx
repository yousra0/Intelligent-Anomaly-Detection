"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { v4 as uuidv4 } from "uuid";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { UploadDropzone } from "@/components/datasets/UploadDropzone";
import { ResultsDashboard } from "./ResultsDashboard";
import { analysisService } from "@/lib/api/analysisService";
import { analysisRunService } from "@/lib/api/analysisRunService";
import { auditLogService } from "@/lib/api/auditLogService";
import { useAuth } from "@/lib/auth/AuthContext";
import {
  CheckCircle2,
  Upload,
  Brain,
  BarChart3,
  Loader2,
  AlertCircle,
  Cpu,
  Zap,
  Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { PredictResponse } from "@/types";
import type { AxiosError } from "axios";

type WizardStep = "upload" | "model" | "run" | "results";
type ModelChoice = "combined" | "xgboost" | "autoencoder";

const STEPS: { key: WizardStep; label: string; icon: typeof Upload }[] = [
  { key: "upload", label: "Dataset", icon: Upload },
  { key: "model", label: "Modèle", icon: Brain },
  { key: "run", label: "Analyse", icon: Cpu },
  { key: "results", label: "Résultats", icon: BarChart3 },
];

const MODELS: { key: ModelChoice; label: string; description: string; icon: typeof Zap; badge?: string }[] = [
  {
    key: "combined",
    label: "Analyse combinée",
    description: "XGBoost + AutoEncoder. Meilleure précision. Recommandé pour les données PaySim.",
    icon: Shield,
    badge: "Recommandé",
  },
  {
    key: "xgboost",
    label: "XGBoost uniquement",
    description: "Modèle supervisé. Recall=0.846, F1=0.835. Rapide et très précis.",
    icon: Zap,
  },
  {
    key: "autoencoder",
    label: "AutoEncoder uniquement",
    description: "Détection non-supervisée. Détecte des fraudes inconnues (zero-day). Recall=0.359.",
    icon: Brain,
  },
];

interface AnalysisWizardProps {
  missionId?: string;
  missionName?: string;
  companyName?: string;
  datasetId?: string;
}

export function AnalysisWizard({ missionId, missionName, companyName, datasetId }: AnalysisWizardProps) {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const [step, setStep] = useState<WizardStep>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedModel, setSelectedModel] = useState<ModelChoice>("combined");
  const [result, setResult] = useState<PredictResponse | null>(null);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: analysisService.health,
    retry: 1,
  });

  const predictMutation = useMutation({
    mutationFn: () => {
      if (!file) throw new Error("Aucun fichier");
      return analysisService.predict(file, setUploadProgress);
    },
    onSuccess: async (data) => {
      setResult(data);
      setStep("results");
      toast.success(`Analyse terminée — ${data.n_fraud} anomalie(s) détectée(s).`);

      // Persist analysis run
      const run = {
        id: uuidv4(),
        mission_id: missionId ?? "",
        mission_name: missionName ?? "Mission inconnue",
        company_name: companyName ?? "",
        dataset_id: datasetId ?? uuidv4(),
        dataset_name: file?.name ?? "dataset",
        model_mode: selectedModel as "combined",
        status: "completed" as const,
        started_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
        result: data,
        user_id: user?.id ?? "u1",
        user_name: user?.name ?? "Inconnu",
      };

      try {
        await analysisRunService.save(run);
        queryClient.invalidateQueries({ queryKey: ["analysis-runs"] });
      } catch {
        // Non-blocking — run persisted locally even if API fails
      }
    },
    onError: async (err: Error) => {
      const axiosErr = err as AxiosError<{ detail?: string | Record<string, unknown> }>;
      const detail = axiosErr.response?.data?.detail;
      const userMsg =
        typeof detail === "string"
          ? detail
          : typeof detail === "object" && detail !== null
          ? (detail as { error?: string }).error ?? JSON.stringify(detail)
          : err.message;
      toast.error(`Erreur d'analyse : ${userMsg}`);

      // Log failure to audit trail
      try {
        await auditLogService.log({
          action: "analysis.start",
          mission_id: missionId,
          mission_name: missionName,
          details: `Échec analyse sur "${file?.name}" — ${err.message}`,
        });
      } catch {
        // non-blocking
      }
    },
  });

  const stepIndex = STEPS.findIndex((s) => s.key === step);

  if (step === "results" && result) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Résultats de l'analyse</h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setStep("upload");
              setFile(null);
              setResult(null);
              setUploadProgress(0);
            }}
          >
            Nouvelle analyse
          </Button>
        </div>
        <ResultsDashboard result={result} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Backend health indicator */}
      {health && (
        <div
          className={cn(
            "flex items-center gap-2 rounded-lg border px-3 py-2 text-xs",
            health.status === "ok"
              ? "border-green-200 bg-green-50 text-green-700"
              : "border-red-200 bg-red-50 text-red-700"
          )}
        >
          <span className={cn("h-2 w-2 rounded-full", health.status === "ok" ? "bg-green-500" : "bg-red-500")} />
          Backend FastAPI{" "}
          {health.status === "ok" ? "opérationnel" : "indisponible"} ·{" "}
          Modèles{" "}{health.models_loaded ? "chargés ✓" : "non chargés ✗"} ·{" "}
          LLM{" "}{health.llm_available ? "disponible ✓" : "indisponible"}
        </div>
      )}

      {/* Step indicators */}
      <div className="flex items-center gap-1">
        {STEPS.filter((s) => s.key !== "results").map((s, i) => {
          const Icon = s.icon;
          const isActive = s.key === step;
          const isDone = STEPS.findIndex((x) => x.key === step) > i;
          return (
            <div key={s.key} className="flex items-center">
              <div
                className={cn(
                  "flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
                  isActive ? "bg-pwc-orange text-white" : isDone ? "bg-green-100 text-green-700" : "bg-muted text-muted-foreground"
                )}
              >
                {isDone ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <Icon className="h-3.5 w-3.5" />
                )}
                {s.label}
              </div>
              {i < 2 && <div className="mx-1 h-px w-6 bg-border" />}
            </div>
          );
        })}
      </div>

      {/* Step 1: Upload */}
      {step === "upload" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Étape 1 — Sélectionner le dataset</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <UploadDropzone
              onFileSelect={(f) => setFile(f)}
              label="Glissez votre CSV de transactions ici"
            />
            {file && (
              <div className="flex justify-end">
                <Button onClick={() => setStep("model")}>Continuer</Button>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Step 2: Model selection */}
      {step === "model" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Étape 2 — Choisir le modèle</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-3">
              {MODELS.map((m) => {
                const Icon = m.icon;
                const isSelected = selectedModel === m.key;
                return (
                  <button
                    key={m.key}
                    className={cn(
                      "relative rounded-lg border-2 p-4 text-left transition-all",
                      isSelected
                        ? "border-pwc-orange bg-orange-50"
                        : "border-border hover:border-pwc-orange/50 hover:bg-accent"
                    )}
                    onClick={() => setSelectedModel(m.key)}
                  >
                    {m.badge && (
                      <span className="absolute right-2 top-2 rounded-full bg-pwc-orange px-2 py-0.5 text-[10px] font-semibold text-white">
                        {m.badge}
                      </span>
                    )}
                    <Icon
                      className={cn(
                        "mb-2 h-6 w-6",
                        isSelected ? "text-pwc-orange" : "text-muted-foreground"
                      )}
                    />
                    <p className="text-sm font-semibold">{m.label}</p>
                    <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                      {m.description}
                    </p>
                  </button>
                );
              })}
            </div>
            <div className="flex justify-between">
              <Button variant="outline" onClick={() => setStep("upload")}>
                Retour
              </Button>
              <Button onClick={() => setStep("run")}>Continuer</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step 3: Run */}
      {step === "run" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Étape 3 — Lancer l'analyse</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Summary */}
            <div className="rounded-lg bg-muted/40 p-4 space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Fichier</span>
                <span className="font-medium">{file?.name}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Modèle</span>
                <Badge variant="default">
                  {MODELS.find((m) => m.key === selectedModel)?.label}
                </Badge>
              </div>
              {missionName && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Mission</span>
                  <span className="font-medium">{missionName}</span>
                </div>
              )}
            </div>

            {predictMutation.isPending ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Analyse en cours via FastAPI…
                </div>
                {uploadProgress > 0 && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Transfert du fichier</span>
                      <span>{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} />
                  </div>
                )}
                <p className="text-xs text-muted-foreground">
                  Pipeline : profilage → mapping sémantique → construction des 14 features → prédiction XGBoost + AutoEncoder…
                </p>
              </div>
            ) : predictMutation.isError ? (
              <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-3">
                <AlertCircle className="h-4 w-4 mt-0.5 text-red-600 shrink-0" />
                <div className="flex-1 text-sm text-red-800">
                  <p className="font-medium">Erreur lors de l'analyse</p>
                  <p className="mt-1 text-xs leading-relaxed">
                    {(() => {
                      const axiosErr = predictMutation.error as AxiosError<{ detail?: string | Record<string, unknown>; error?: string }>;
                      const res = axiosErr?.response;
                      const detail = res?.data?.detail;
                      if (typeof detail === "string") return detail;
                      if (typeof detail === "object" && detail !== null) {
                        const obj = detail as { error?: string; columns_in_csv?: string[] };
                        return obj.error
                          ? `${obj.error}${obj.columns_in_csv ? ` — Colonnes détectées : ${obj.columns_in_csv.join(", ")}` : ""}`
                          : JSON.stringify(detail, null, 2);
                      }
                      // Pas de réponse = réseau / CORS
                      if (!res) {
                        return "Impossible de joindre le backend FastAPI. Vérifiez qu'il est démarré sur le port attendu.";
                      }
                      // 500 sans detail = proxy Next.js → backend non démarré
                      if (res.status === 500 && !detail) {
                        return "Le backend FastAPI n'est pas démarré (port 8000). Lancez : uvicorn app.main:app --reload --port 8000";
                      }
                      return axiosErr?.message ?? "Erreur inconnue — vérifiez la console du navigateur.";
                    })()}
                  </p>
                  <p className="mt-2 text-xs text-red-600">
                    Le système accepte n'importe quel fichier CSV : PaySim, grand livre, balance, transactions ERP…
                    Aucune colonne spécifique n'est requise.
                  </p>
                </div>
              </div>
            ) : null}

            <div className="flex justify-between">
              <Button
                variant="outline"
                onClick={() => setStep("model")}
                disabled={predictMutation.isPending}
              >
                Retour
              </Button>
              <Button
                onClick={() => predictMutation.mutate()}
                disabled={predictMutation.isPending}
                className="gap-2"
              >
                {predictMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Cpu className="h-4 w-4" />
                )}
                {predictMutation.isPending ? "Analyse en cours…" : "Lancer l'analyse"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
