"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { KPICards } from "./KPICards";
import { AnomalyTable } from "./AnomalyTable";
import { RiskPieChart } from "./charts/RiskPieChart";
import { AnomalyBarChart } from "./charts/AnomalyBarChart";
import { ScoreDistributionChart } from "./charts/ScoreDistributionChart";
import { ExplanationCard } from "@/components/explanations/ExplanationCard";
import { ReportSection } from "@/components/reports/ReportSection";
import { llmService } from "@/lib/api/llmService";
import { Loader2, Info } from "lucide-react";
import type { PredictResponse, ExplainResponse } from "@/types";

interface ResultsDashboardProps {
  result: PredictResponse;
}

export function ResultsDashboard({ result }: ResultsDashboardProps) {
  const [selectedTxId, setSelectedTxId] = useState<number | null>(null);

  const { data: explanation, isLoading: explainLoading } = useQuery({
    queryKey: ["explain", selectedTxId],
    queryFn: () => llmService.getExplanation(selectedTxId!),
    enabled: selectedTxId !== null,
  });

  const fraudTxIds = result.transactions
    .filter((t) => t.is_fraud_predicted)
    .map((t) => t.tx_id)
    .slice(0, 20);

  const handleExplain = (txId: number) => {
    setSelectedTxId(txId);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* KPI Cards */}
      <KPICards result={result} />

      {/* Schema detection info banner */}
      {result.schema_detection.warnings.length > 0 && (
        <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-3">
          <Info className="h-4 w-4 mt-0.5 shrink-0 text-amber-600" />
          <div className="text-sm text-amber-800">
            <p className="font-medium">Avertissements de détection de schéma</p>
            <ul className="mt-1 list-disc pl-4 text-xs space-y-0.5">
              {result.schema_detection.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Mode used */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span className="rounded bg-muted px-2 py-0.5 text-xs font-mono">
          mode: {result.prediction_mode}
        </span>
        <span>—</span>
        <span>{result.schema_detection.reason}</span>
      </div>

      <Tabs defaultValue="charts">
        <TabsList>
          <TabsTrigger value="charts">Graphiques</TabsTrigger>
          <TabsTrigger value="table">
            Transactions ({result.n_transactions.toLocaleString("fr-FR")})
          </TabsTrigger>
          {selectedTxId !== null && (
            <TabsTrigger value="explain">
              Explication #{selectedTxId}
            </TabsTrigger>
          )}
          <TabsTrigger value="report">Rapport</TabsTrigger>
        </TabsList>

        {/* Charts tab */}
        <TabsContent value="charts" className="space-y-4 mt-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Répartition Normal / Anomalie
                </CardTitle>
              </CardHeader>
              <CardContent>
                <RiskPieChart result={result} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  Distribution des scores
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScoreDistributionChart
                  transactions={result.transactions}
                  threshold={result.threshold}
                />
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">
                Analyse par catégorie
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AnomalyBarChart transactions={result.transactions} />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Table tab */}
        <TabsContent value="table" className="mt-4">
          <AnomalyTable
            transactions={result.transactions}
            onExplain={handleExplain}
          />
        </TabsContent>

        {/* Explanation tab */}
        {selectedTxId !== null && (
          <TabsContent value="explain" className="mt-4">
            {explainLoading ? (
              <div className="flex items-center justify-center py-16 gap-2 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin" />
                Génération de l'explication LLM en cours…
              </div>
            ) : explanation ? (
              <ExplanationCard explanation={explanation} />
            ) : null}
          </TabsContent>
        )}

        {/* Report tab */}
        <TabsContent value="report" className="mt-4">
          <ReportSection />
        </TabsContent>
      </Tabs>
    </div>
  );
}
