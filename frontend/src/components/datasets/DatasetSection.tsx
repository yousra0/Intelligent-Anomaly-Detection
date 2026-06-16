"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { UploadDropzone } from "./UploadDropzone";
import {
  FileText,
  Upload,
  RefreshCw,
  Trash2,
  BarChart3,
  Plus,
  Loader2,
} from "lucide-react";
import {
  formatFileSize,
  formatDateTime,
  DATASET_CATEGORY_LABELS_FR,
} from "@/lib/utils";
import { datasetService } from "@/lib/api/datasetService";
import type { Dataset, DatasetCategory } from "@/types";

interface DatasetSectionProps {
  missionId: string;
  onAnalyze?: (dataset: Dataset) => void;
}

export function DatasetSection({ missionId, onAnalyze }: DatasetSectionProps) {
  const queryClient = useQueryClient();
  const [uploadOpen, setUploadOpen] = useState(false);
  const [replaceTarget, setReplaceTarget] = useState<Dataset | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadCategory, setUploadCategory] = useState<DatasetCategory>("transactions");
  const [uploadProgress, setUploadProgress] = useState(0);

  const { data: datasets = [], isLoading } = useQuery({
    queryKey: ["datasets", missionId],
    queryFn: () => datasetService.getByMission(missionId),
  });

  const uploadMutation = useMutation({
    mutationFn: () => {
      if (!uploadFile) throw new Error("Aucun fichier sélectionné");
      return datasetService.upload(missionId, uploadFile, uploadCategory, setUploadProgress);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", missionId] });
      toast.success("Dataset importé avec succès.");
      setUploadOpen(false);
      setUploadFile(null);
      setUploadProgress(0);
    },
    onError: () => toast.error("Erreur lors de l'import du dataset."),
  });

  const replaceMutation = useMutation({
    mutationFn: () => {
      if (!replaceTarget || !uploadFile) throw new Error();
      return datasetService.replace(missionId, replaceTarget.id, uploadFile, setUploadProgress);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", missionId] });
      toast.success("Dataset remplacé.");
      setReplaceTarget(null);
      setUploadFile(null);
      setUploadProgress(0);
    },
    onError: () => toast.error("Erreur lors du remplacement."),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => datasetService.delete(missionId, id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets", missionId] });
      toast.success("Dataset supprimé.");
    },
    onError: () => toast.error("Erreur lors de la suppression."),
  });

  const CATEGORIES: DatasetCategory[] = ["transactions", "general_ledger", "trial_balance"];

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground py-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        Chargement des datasets…
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Datasets financiers</h2>
        <Button size="sm" className="gap-2" onClick={() => setUploadOpen(true)}>
          <Plus className="h-4 w-4" />
          Importer un dataset
        </Button>
      </div>

      {/* Category sections */}
      {CATEGORIES.map((cat) => {
        const catDatasets = datasets.filter((d) => d.category === cat);
        return (
          <div key={cat}>
            <h3 className="mb-2 text-sm font-medium text-muted-foreground">
              {DATASET_CATEGORY_LABELS_FR[cat]}
            </h3>
            {catDatasets.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border bg-muted/20 p-6 text-center">
                <p className="text-sm text-muted-foreground mb-2">
                  Aucun dataset de type «{DATASET_CATEGORY_LABELS_FR[cat]}» importé.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-2"
                  onClick={() => {
                    setUploadCategory(cat);
                    setUploadOpen(true);
                  }}
                >
                  <Upload className="h-4 w-4" />
                  Importer
                </Button>
              </div>
            ) : (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {catDatasets.map((ds) => (
                  <Card key={ds.id} className="flex flex-col">
                    <CardHeader className="pb-2">
                      <div className="flex items-start gap-2">
                        <FileText className="mt-0.5 h-4 w-4 shrink-0 text-pwc-orange" />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium truncate" title={ds.name}>
                            {ds.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {formatFileSize(ds.file_size)} · {formatDateTime(ds.uploaded_at)}
                          </p>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="flex-1 space-y-3">
                      <Badge variant={ds.status === "analyzed" ? "faible" : ds.status === "error" ? "critique" : "outline"}>
                        {ds.status === "uploaded" && "Importé"}
                        {ds.status === "analyzing" && "En analyse…"}
                        {ds.status === "analyzed" && "Analysé"}
                        {ds.status === "error" && "Erreur"}
                        {ds.status === "pending" && "En attente"}
                      </Badge>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1 gap-1 text-xs"
                          onClick={() => { setReplaceTarget(ds); setUploadFile(null); }}
                        >
                          <RefreshCw className="h-3 w-3" />
                          Remplacer
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-1 text-xs text-green-700 border-green-200 hover:bg-green-50"
                          onClick={() => onAnalyze?.(ds)}
                        >
                          <BarChart3 className="h-3 w-3" />
                          Analyser
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="gap-1 text-xs text-red-600 border-red-200 hover:bg-red-50"
                          onClick={() => deleteMutation.mutate(ds.id)}
                          disabled={deleteMutation.isPending}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        );
      })}

      {/* Upload dialog */}
      <Dialog open={uploadOpen} onOpenChange={(o) => !o && setUploadOpen(false)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Importer un dataset</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label>Catégorie</Label>
              <Select
                value={uploadCategory}
                onValueChange={(v) => setUploadCategory(v as DatasetCategory)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {CATEGORIES.map((c) => (
                    <SelectItem key={c} value={c}>
                      {DATASET_CATEGORY_LABELS_FR[c]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <UploadDropzone
              onFileSelect={setUploadFile}
              uploadProgress={uploadMutation.isPending ? uploadProgress : undefined}
              disabled={uploadMutation.isPending}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setUploadOpen(false)}>
              Annuler
            </Button>
            <Button
              disabled={!uploadFile || uploadMutation.isPending}
              onClick={() => uploadMutation.mutate()}
            >
              {uploadMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Importer
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Replace dialog */}
      <Dialog open={!!replaceTarget} onOpenChange={(o) => !o && setReplaceTarget(null)}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Remplacer {replaceTarget?.name}</DialogTitle>
          </DialogHeader>
          <UploadDropzone
            onFileSelect={setUploadFile}
            uploadProgress={replaceMutation.isPending ? uploadProgress : undefined}
            disabled={replaceMutation.isPending}
            label="Glissez le nouveau fichier ici"
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setReplaceTarget(null)}>
              Annuler
            </Button>
            <Button
              disabled={!uploadFile || replaceMutation.isPending}
              onClick={() => replaceMutation.mutate()}
            >
              {replaceMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Remplacer
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
