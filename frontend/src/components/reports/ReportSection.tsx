"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { reportService } from "@/lib/api/reportService";
import {
  FileText,
  FileSpreadsheet,
  Download,
  RefreshCw,
  Loader2,
  CheckCircle2,
} from "lucide-react";
import { formatDateTime } from "@/lib/utils";

interface GeneratedReport {
  format: "pdf" | "docx";
  name: string;
  generatedAt: string;
  blob: Blob;
}

export function ReportSection() {
  const [reports, setReports] = useState<GeneratedReport[]>([]);

  const pdfMutation = useMutation({
    mutationFn: reportService.generatePDF,
    onSuccess: (blob) => {
      const name = `Rapport_Fraude_PwC_${new Date().toISOString().slice(0, 10)}.pdf`;
      setReports((prev) => [
        { format: "pdf", name, generatedAt: new Date().toISOString(), blob },
        ...prev.filter((r) => r.format !== "pdf"),
      ]);
      reportService.downloadBlob(blob, name);
      toast.success("Rapport PDF généré et téléchargé.");
    },
    onError: () => toast.error("Erreur lors de la génération du PDF."),
  });

  const docxMutation = useMutation({
    mutationFn: reportService.generateDOCX,
    onSuccess: (blob) => {
      const name = `Rapport_Fraude_PwC_${new Date().toISOString().slice(0, 10)}.docx`;
      setReports((prev) => [
        { format: "docx", name, generatedAt: new Date().toISOString(), blob },
        ...prev.filter((r) => r.format !== "docx"),
      ]);
      reportService.downloadBlob(blob, name);
      toast.success("Rapport Word généré et téléchargé.");
    },
    onError: () => toast.error("Erreur lors de la génération du rapport Word."),
  });

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        {/* PDF Card */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <FileText className="h-4 w-4 text-red-500" />
              Rapport PDF
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground leading-relaxed">
              Rapport complet à la charte PwC : page de couverture, résumé exécutif, 3 graphiques,
              tableau top-10, cartes CRITIQUE avec facteurs SHAP, recommandations, glossaire.
            </p>
            <div className="space-y-1.5 text-xs text-muted-foreground">
              {[
                "Page de couverture avec jauge de risque",
                "4 KPI + 3 graphiques matplotlib",
                "Cartes détaillées (max 8 CRITIQUE)",
                "Recommandations + glossaire",
              ].map((item) => (
                <div key={item} className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
            <Button
              className="w-full gap-2"
              onClick={() => pdfMutation.mutate()}
              disabled={pdfMutation.isPending}
            >
              {pdfMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <FileText className="h-4 w-4" />
              )}
              {pdfMutation.isPending ? "Génération…" : "Générer le PDF"}
            </Button>
          </CardContent>
        </Card>

        {/* DOCX Card */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <FileSpreadsheet className="h-4 w-4 text-blue-500" />
              Rapport Word (DOCX)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground leading-relaxed">
              Rapport Word depuis le template PwC avec placeholders dynamiques : 5 graphiques
              intégrés, tableau des top-10, cartes CRITIQUE (max 14), sections recommandations.
            </p>
            <div className="space-y-1.5 text-xs text-muted-foreground">
              {[
                "Template Word PwC (exemple_rapport.docx)",
                "5 graphiques matplotlib intégrés",
                "Cartes détaillées (max 14 CRITIQUE)",
                "Traductions FR des features",
              ].map((item) => (
                <div key={item} className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500 shrink-0" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
            <Button
              className="w-full gap-2"
              variant="outline"
              onClick={() => docxMutation.mutate()}
              disabled={docxMutation.isPending}
            >
              {docxMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <FileSpreadsheet className="h-4 w-4" />
              )}
              {docxMutation.isPending ? "Génération…" : "Générer le DOCX"}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Generated reports list */}
      {reports.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-medium text-muted-foreground">Rapports générés</h3>
          <div className="space-y-2">
            {reports.map((r) => (
              <div
                key={r.format}
                className="flex items-center justify-between rounded-lg border bg-white p-3"
              >
                <div className="flex items-center gap-3">
                  {r.format === "pdf" ? (
                    <FileText className="h-5 w-5 text-red-500" />
                  ) : (
                    <FileSpreadsheet className="h-5 w-5 text-blue-500" />
                  )}
                  <div>
                    <p className="text-sm font-medium">{r.name}</p>
                    <p className="text-xs text-muted-foreground">
                      Généré le {formatDateTime(r.generatedAt)}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="gap-1 text-xs"
                    onClick={() => reportService.downloadBlob(r.blob, r.name)}
                  >
                    <Download className="h-3.5 w-3.5" />
                    Télécharger
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="gap-1 text-xs"
                    onClick={() => r.format === "pdf" ? pdfMutation.mutate() : docxMutation.mutate()}
                    disabled={pdfMutation.isPending || docxMutation.isPending}
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Régénérer
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
