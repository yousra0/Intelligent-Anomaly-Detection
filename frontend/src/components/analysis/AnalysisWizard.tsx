"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import { toast } from "sonner";
import { v4 as uuidv4 } from "uuid";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { UploadDropzone } from "@/components/datasets/UploadDropzone";
import { analysisService } from "@/lib/api/analysisService";
import { analysisRunService } from "@/lib/api/analysisRunService";
import { datasetService } from "@/lib/api/datasetService";
import {
  CheckCircle2,
  Upload,
  BarChart3,
  Loader2,
  AlertCircle,
  Cpu,
  Shield,
  FileText,
  Clock,
  ChevronRight,
} from "lucide-react";
import { cn, formatFileSize, formatDateTime } from "@/lib/utils";
import type { PredictResponse, Dataset, AnalysisRunRecord } from "@/types";
import type { AxiosError } from "axios";

const ResultsDashboard = dynamic(
  () => import("./ResultsDashboard").then((m) => ({ default: m.ResultsDashboard })),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center py-16 text-sm text-muted-foreground gap-2">
        <Loader2 className="h-4 w-4 animate-spin" />
        Chargement des résultats…
      </div>
    ),
  }
);

type WizardStep = "upload" | "run" | "results";

const STEPS: { key: WizardStep; label: string; icon: typeof Upload }[] = [
  { key: "upload", label: "Dataset", icon: Upload },
  { key: "run", label: "Analyse", icon: Cpu },
  { key: "results", label: "Résultats", icon: BarChart3 },
];

interface AnalysisWizardProps {
  missionId?: string;
  missionName?: string;
  companyName?: string;
  /** Pre-selected dataset ID (e.g. passed via URL param from mission page) */
  datasetId?: string;
}

export function AnalysisWizard({ missionId, missionName, companyName, datasetId: initialDatasetId }: AnalysisWizardProps) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState<WizardStep>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [activeDatasetId, setActiveDatasetId] = useState<string | undefined>(initialDatasetId);
  // Track the run ID of the result being viewed (for "Voir" on past runs)
  const [viewedRunId, setViewedRunId] = useState<string | undefined>();

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: analysisService.health,
    retry: 1,
    staleTime: 30_000,
  });

  const { data: existingDatasets = [] } = useQuery<Dataset[]>({
    queryKey: ["datasets", missionId],
    queryFn: () => datasetService.getByMission(missionId!),
    enabled: !!missionId,
  });

  const { data: pastRuns = [] } = useQuery<AnalysisRunRecord[]>({
    queryKey: ["analysis-runs", missionId],
    queryFn: () => analysisRunService.getByMission(missionId!),
    enabled: !!missionId,
  });

  /**
   * Mutation principale du wizard — orchestre le pipeline en deux temps :
   *   1. Persistance du dataset : si le fichier est nouveau (pas de dsId), on
   *      crée d'abord un enregistrement en base via Next.js pour que la FK
   *      dataset_id soit valide avant d'envoyer à FastAPI.
   *   2. Prédiction : POST multipart/form-data vers FastAPI /api/predict.
   *      Le callback de progression met à jour la barre de téléchargement.
   * En cas de succès, l'AnalysisRun est sauvegardé en base (non-bloquant).
   */
  const analyzeMutation = useMutation({
    mutationFn: async (input: { file: File; dsId?: string }) => {
      let resolvedDatasetId = input.dsId;

      // New file → persist as a dataset record first so FK is valid
      if (!resolvedDatasetId && missionId) {
        const ds = await datasetService.upload(missionId, input.file, "transactions");
        resolvedDatasetId = ds.id;
        setActiveDatasetId(resolvedDatasetId);
        queryClient.invalidateQueries({ queryKey: ["datasets", missionId] });
      }

      const prediction = await analysisService.predict(input.file, setUploadProgress);
      return { prediction, resolvedDatasetId };
    },
    onSuccess: async ({ prediction, resolvedDatasetId }, { file: fileVar }) => {
      setResult(prediction);
      setStep("results");
      toast.success(`Analyse terminée — ${prediction.n_fraud} anomalie(s) détectée(s).`);

      if (!missionId) return;
      try {
        // Sauvegarde l'analyse en base pour qu'elle apparaisse dans l'historique
        const saved = await analysisRunService.save({
          id: uuidv4(),
          mission_id: missionId,
          mission_name: missionName ?? "Mission inconnue",
          company_name: companyName ?? "",
          dataset_id: resolvedDatasetId ?? uuidv4(),
          dataset_name: fileVar.name,
          model_mode: "combined",
          status: "completed",
          started_at: new Date().toISOString(),
          completed_at: new Date().toISOString(),
          result: prediction,
        });
        setViewedRunId(saved.id);
        queryClient.invalidateQueries({ queryKey: ["analysis-runs", missionId] });
      } catch {
        // non-blocking
      }
    },
    onError: (err: Error) => {
      // FastAPI renvoie parfois { detail: { error: "...", columns_in_csv: [...] } }
      // Il faut extraire le message lisible depuis la structure Axios imbriquée
      const axiosErr = err as AxiosError<{ detail?: string | Record<string, unknown> }>;
      const detail = axiosErr.response?.data?.detail;
      const userMsg =
        typeof detail === "string"
          ? detail
          : typeof detail === "object" && detail !== null
          ? (detail as { error?: string }).error ?? JSON.stringify(detail)
          : err.message;
      toast.error(`Erreur d'analyse : ${userMsg}`);
    },
  });

  const handleFileSelect = (f: File) => {
    setFile(f);
    setUploadProgress(0);
    setStep("run");
    analyzeMutation.mutate({ file: f }); // no dsId → will save as dataset first
  };

  const handleDatasetSelect = async (ds: Dataset) => {
    setStep("run");
    setActiveDatasetId(ds.id);
    try {
      toast.info(`Chargement de « ${ds.name } »…`);
      const fileObj = await datasetService.getContent(missionId!, ds.id);
      setFile(fileObj);
      analyzeMutation.mutate({ file: fileObj, dsId: ds.id });
    } catch {
      toast.error("Impossible de charger le fichier. Réimportez-le depuis la page de la mission.");
      setStep("upload");
    }
  };

  const handleViewRun = (run: AnalysisRunRecord) => {
    if (run.result) {
      setResult(run.result as PredictResponse);
      setViewedRunId(run.id);
      setStep("results");
    }
  };

  const reset = () => {
    setStep("upload");
    setFile(null);
    setResult(null);
    setUploadProgress(0);
    setActiveDatasetId(initialDatasetId);
    analyzeMutation.reset();
  };

  const stepIndex = STEPS.findIndex((s) => s.key === step);

  if (step === "results" && result) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Résultats de l'analyse</h2>
          <Button variant="outline" size="sm" onClick={reset}>
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
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              health.status === "ok" ? "bg-green-500" : "bg-red-500"
            )}
          />
          Backend FastAPI {health.status === "ok" ? "opérationnel" : "indisponible"} ·{" "}
          Modèles {health.models_loaded ? "chargés ✓" : "non chargés ✗"} ·{" "}
          LLM {health.llm_available ? "disponible ✓" : "indisponible"}
        </div>
      )}

      {/* Step indicators */}
      <div className="flex items-center gap-1">
        {STEPS.filter((s) => s.key !== "results").map((s, i, arr) => {
          const Icon = s.icon;
          const sIndex = STEPS.findIndex((x) => x.key === s.key);
          const isActive = s.key === step;
          const isDone = stepIndex > sIndex;
          return (
            <div key={s.key} className="flex items-center">
              <div
                className={cn(
                  "flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
                  isActive
                    ? "bg-pwc-orange text-white"
                    : isDone
                    ? "bg-green-100 text-green-700"
                    : "bg-muted text-muted-foreground"
                )}
              >
                {isDone ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <Icon className="h-3.5 w-3.5" />
                )}
                {s.label}
              </div>
              {i < arr.length - 1 && <div className="mx-1 h-px w-6 bg-border" />}
            </div>
          );
        })}
      </div>

      {/* Step 1: Upload / Select */}
      {step === "upload" && (
        <div className="space-y-4">
          {/* Existing datasets */}
          {existingDatasets.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <FileText className="h-4 w-4 text-pwc-orange" />
                  Fichiers déjà importés pour cette mission
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {existingDatasets.map((ds) => (
                  <div
                    key={ds.id}
                    className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2.5 hover:bg-muted/60 transition-colors"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <FileText className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">{ds.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatFileSize(ds.file_size)} · importé le {formatDateTime(ds.uploaded_at)}
                        </p>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      className="ml-3 gap-1.5 shrink-0"
                      onClick={() => handleDatasetSelect(ds)}
                      disabled={analyzeMutation.isPending}
                    >
                      <Cpu className="h-3.5 w-3.5" />
                      Analyser
                      <ChevronRight className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* New file upload */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {existingDatasets.length > 0
                  ? "Ou importer un nouveau fichier"
                  : "Étape 1 — Importer le dataset"}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 px-3 py-2 text-xs text-orange-800">
                <Shield className="h-3.5 w-3.5 shrink-0 text-pwc-orange" />
                <span>
                  Analyse combinée{" "}
                  <strong>XGBoost + AutoEncoder</strong> — meilleure précision
                  pour la détection de fraudes. L'analyse démarre automatiquement à l'import.
                </span>
              </div>
              <UploadDropzone
                onFileSelect={handleFileSelect}
                label="Glissez votre CSV de transactions ici"
              />
            </CardContent>
          </Card>

          {/* Past analyses */}
          {pastRuns.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  Analyses précédentes
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {pastRuns.slice(0, 5).map((run) => (
                  <div
                    key={run.id}
                    className="flex items-center justify-between rounded-lg border px-3 py-2.5"
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <BarChart3 className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <div className="min-w-0">
                        <p className="text-sm font-medium truncate">{run.dataset_name}</p>
                        <p className="text-xs text-muted-foreground">
                          {run.completed_at ? formatDateTime(run.completed_at) : "—"}
                          {run.result && ` · ${(run.result as PredictResponse).n_fraud} anomalie(s)`}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 ml-3 shrink-0">
                      <Badge
                        variant={
                          run.status === "completed"
                            ? "faible"
                            : run.status === "error"
                            ? "critique"
                            : "outline"
                        }
                        className="text-xs"
                      >
                        {run.status === "completed" ? "Terminée" : run.status === "error" ? "Erreur" : run.status}
                      </Badge>
                      {run.result && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="text-xs gap-1"
                          onClick={() => handleViewRun(run)}
                        >
                          <BarChart3 className="h-3 w-3" />
                          Voir
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Step 2: Analysis running */}
      {step === "run" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Étape 2 — Analyse en cours</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="rounded-lg bg-muted/40 p-4 space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Fichier</span>
                <span className="font-medium">{file?.name}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Modèle</span>
                <Badge variant="default">XGBoost + AutoEncoder</Badge>
              </div>
              {missionName && (
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Mission</span>
                  <span className="font-medium">{missionName}</span>
                </div>
              )}
            </div>

            {analyzeMutation.isPending && (
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
                  Pipeline : profilage → mapping sémantique → construction des
                  14 features → prédiction XGBoost + AutoEncoder…
                </p>
              </div>
            )}

            {analyzeMutation.isError && (
              <div className="space-y-3">
                <div className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-3">
                  <AlertCircle className="h-4 w-4 mt-0.5 text-red-600 shrink-0" />
                  <div className="flex-1 text-sm text-red-800">
                    <p className="font-medium">Erreur lors de l'analyse</p>
                    <p className="mt-1 text-xs leading-relaxed">
                      {(() => {
                        const axiosErr = analyzeMutation.error as AxiosError<{
                          detail?: string | Record<string, unknown>;
                          error?: string;
                        }>;
                        const res = axiosErr?.response;
                        const detail = res?.data?.detail;
                        if (typeof detail === "string") return detail;
                        if (typeof detail === "object" && detail !== null) {
                          const obj = detail as { error?: string; columns_in_csv?: string[] };
                          return obj.error
                            ? `${obj.error}${obj.columns_in_csv ? ` — Colonnes détectées : ${obj.columns_in_csv.join(", ")}` : ""}`
                            : JSON.stringify(detail, null, 2);
                        }
                        if (!res)
                          return "Impossible de joindre le backend FastAPI. Vérifiez qu'il est démarré sur le port attendu.";
                        if (res.status === 500 && !detail)
                          return "Le backend FastAPI n'est pas démarré (port 8000). Lancez : uvicorn app.main:app --reload --port 8000";
                        return axiosErr?.message ?? "Erreur inconnue — vérifiez la console du navigateur.";
                      })()}
                    </p>
                    <p className="mt-2 text-xs text-red-600">
                      Le système accepte n'importe quel fichier CSV : grand livre,
                      balance, transactions ERP, export bancaire… Aucune colonne
                      spécifique n'est requise.
                    </p>
                  </div>
                </div>
                <div className="flex justify-between">
                  <Button variant="outline" onClick={reset}>
                    Recommencer
                  </Button>
                  <Button
                    onClick={() => file && analyzeMutation.mutate({ file, dsId: activeDatasetId })}
                    className="gap-2"
                  >
                    <Cpu className="h-4 w-4" />
                    Réessayer
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
