"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { analysisRunService } from "@/lib/api/analysisRunService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  formatDate,
  formatDateTime,
  formatPercent,
} from "@/lib/utils";
import {
  History,
  Search,
  BarChart3,
  Building2,
  Calendar,
  Brain,
  AlertTriangle,
  Download,
} from "lucide-react";
import type { AnalysisRunRecord } from "@/types";

const MODEL_LABELS: Record<string, string> = {
  combined:    "Combiné (XGB + AE)",
  xgboost:     "XGBoost",
  autoencoder: "AutoEncoder",
  paysim:      "PaySim",
  ae_isoforest:"AE + IsoForest",
  ae_only:     "AutoEncoder seul",
  isoforest:   "IsoForest",
};

function exportCSV(runs: AnalysisRunRecord[]) {
  const headers = ["Mission", "Société", "Dataset", "Modèle", "Transactions", "Anomalies", "Taux fraude", "Montant à risque", "Date"];
  const rows = runs.map((r) => [
    r.mission_name,
    r.company_name,
    r.dataset_name,
    MODEL_LABELS[r.model_mode] ?? r.model_mode,
    r.result?.n_transactions ?? 0,
    r.result?.n_fraud ?? 0,
    r.result?.fraud_rate_pct?.toFixed(2) ?? "0",
    r.result?.amount_at_risk?.toFixed(0) ?? "0",
    formatDateTime(r.completed_at ?? r.started_at),
  ]);

  const csv = [headers, ...rows].map((row) => row.join(";")).join("\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `historique_analyses_${formatDate(new Date().toISOString())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function HistoryPage() {
  const [search, setSearch] = useState("");

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ["analysis-runs"],
    queryFn: analysisRunService.getAll,
  });

  const filtered = runs.filter(
    (r) =>
      !search ||
      r.mission_name.toLowerCase().includes(search.toLowerCase()) ||
      r.company_name.toLowerCase().includes(search.toLowerCase()) ||
      r.dataset_name.toLowerCase().includes(search.toLowerCase())
  );

  const totalAnomalies = runs.reduce((acc, r) => acc + (r.result?.n_fraud ?? 0), 0);
  const avgFraudRate =
    runs.length > 0
      ? runs.reduce((acc, r) => acc + (r.result?.fraud_rate_pct ?? 0), 0) / runs.length
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Historique d'analyses</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            {runs.length} analyse{runs.length !== 1 ? "s" : ""} — {totalAnomalies} anomalie(s) totales
          </p>
        </div>
        <Button
          variant="outline"
          className="gap-2 self-start"
          onClick={() => exportCSV(filtered)}
          disabled={filtered.length === 0}
        >
          <Download className="h-4 w-4" />
          Exporter CSV
        </Button>
      </div>

      {/* Summary KPIs */}
      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Analyses totales</p>
            <p className="mt-1 text-3xl font-bold">{runs.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Total anomalies détectées</p>
            <p className="mt-1 text-3xl font-bold text-red-600">{totalAnomalies}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <p className="text-xs text-muted-foreground">Taux de fraude moyen</p>
            <p className="mt-1 text-3xl font-bold text-orange-600">{formatPercent(avgFraudRate)}</p>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Rechercher par mission, société, dataset…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Results */}
      {isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-lg" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <History className="h-12 w-12 text-muted-foreground/30" />
            <p className="mt-4 font-medium text-muted-foreground">
              {search ? "Aucun résultat pour cette recherche." : "Aucune analyse enregistrée."}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Lancez une analyse depuis une mission pour voir l'historique.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-3">
          {filtered.map((run) => (
            <AnalysisRunCard key={run.id} run={run} />
          ))}
        </div>
      )}
    </div>
  );
}

function AnalysisRunCard({ run }: { run: AnalysisRunRecord }) {
  const fraudRate = run.result?.fraud_rate_pct ?? 0;
  const nFraud = run.result?.n_fraud ?? 0;
  const nTx = run.result?.n_transactions ?? 0;

  return (
    <Card className="hover:shadow-sm transition-shadow">
      <CardContent className="py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          {/* Left info */}
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-semibold text-sm">{run.mission_name}</h3>
              <Badge variant="outline" className="text-[10px]">
                <Brain className="mr-1 h-2.5 w-2.5" />
                {MODEL_LABELS[run.model_mode] ?? run.model_mode}
              </Badge>
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <Building2 className="h-3 w-3" /> {run.company_name}
              </span>
              <span className="flex items-center gap-1">
                <BarChart3 className="h-3 w-3" /> {run.dataset_name}
              </span>
              <span className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                {formatDateTime(run.completed_at ?? run.started_at)}
              </span>
              <span className="text-muted-foreground/70">par {run.user_name}</span>
            </div>
          </div>

          {/* Right KPIs */}
          {run.result && (
            <div className="flex items-center gap-4 text-right shrink-0">
              <div>
                <p className="text-[10px] text-muted-foreground">Transactions</p>
                <p className="font-semibold text-sm">{nTx.toLocaleString("fr-FR")}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Anomalies</p>
                <p className={`font-semibold text-sm ${nFraud > 0 ? "text-red-600" : "text-green-600"}`}>
                  {nFraud}
                </p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground">Taux</p>
                <p className={`font-semibold text-sm ${fraudRate > 5 ? "text-red-600" : fraudRate > 1 ? "text-orange-600" : "text-green-600"}`}>
                  {formatPercent(fraudRate)}
                </p>
              </div>
              {nFraud > 0 && (
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-red-100">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
