"use client";

import { useState, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Search, ChevronLeft, ChevronRight, Download, Eye } from "lucide-react";
import { cn, formatScore, RISK_LABELS_FR } from "@/lib/utils";
import type { TransactionResult, RiskLevel } from "@/types";

interface AnomalyTableProps {
  transactions: TransactionResult[];
  onExplain?: (txId: number) => void;
}

const PAGE_SIZES = [10, 25, 50] as const;

export function AnomalyTable({ transactions, onExplain }: AnomalyTableProps) {
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState<RiskLevel | "all">("all");
  const [fraudFilter, setFraudFilter] = useState<"all" | "fraud" | "normal">("all");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState<(typeof PAGE_SIZES)[number]>(10);

  const filtered = useMemo(() => {
    return transactions.filter((t) => {
      if (search && !t.tx_id.toString().includes(search)) return false;
      if (riskFilter !== "all" && t.risk_level !== riskFilter) return false;
      if (fraudFilter === "fraud" && !t.is_fraud_predicted) return false;
      if (fraudFilter === "normal" && t.is_fraud_predicted) return false;
      return true;
    });
  }, [transactions, search, riskFilter, fraudFilter]);

  const totalPages = Math.ceil(filtered.length / pageSize);
  const paginated = filtered.slice(page * pageSize, (page + 1) * pageSize);

  const exportCSV = () => {
    const headers = ["tx_id", "type", "amount", "xgb_score", "ae_score", "risk_level", "is_fraud"];
    const rows = filtered.map((t) => [
      t.tx_id,
      t.type ?? "",
      t.amount ?? "",
      t.xgb_score ?? "",
      t.ae_score ?? "",
      t.risk_level,
      t.is_fraud_predicted,
    ]);
    const csv = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "anomalies_export.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const RISK_BADGE_VARIANT: Record<RiskLevel, "critique" | "eleve" | "faible"> = {
    CRITIQUE: "critique",
    ELEVE: "eleve",
    FAIBLE: "faible",
  };

  return (
    <div className="space-y-3">
      {/* Filters row */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[160px]">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Rechercher par ID…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            className="pl-8 h-8 text-sm"
          />
        </div>

        <Select value={riskFilter} onValueChange={(v) => { setRiskFilter(v as RiskLevel | "all"); setPage(0); }}>
          <SelectTrigger className="h-8 w-[140px] text-sm">
            <SelectValue placeholder="Niveau de risque" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Tous les niveaux</SelectItem>
            <SelectItem value="CRITIQUE">Critique</SelectItem>
            <SelectItem value="ELEVE">Élevé</SelectItem>
            <SelectItem value="FAIBLE">Faible</SelectItem>
          </SelectContent>
        </Select>

        <Select value={fraudFilter} onValueChange={(v) => { setFraudFilter(v as "all" | "fraud" | "normal"); setPage(0); }}>
          <SelectTrigger className="h-8 w-[130px] text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Toutes</SelectItem>
            <SelectItem value="fraud">Fraudes seulement</SelectItem>
            <SelectItem value="normal">Normales seulement</SelectItem>
          </SelectContent>
        </Select>

        <Button variant="outline" size="sm" className="h-8 gap-1 text-xs" onClick={exportCSV}>
          <Download className="h-3.5 w-3.5" />
          Exporter CSV
        </Button>
      </div>

      {/* Table */}
      <div className="rounded-lg border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/40">
              <TableHead className="w-20">ID</TableHead>
              <TableHead>Type</TableHead>
              <TableHead className="text-right">Montant</TableHead>
              <TableHead className="text-right">Score XGB</TableHead>
              <TableHead className="text-right">Score AE</TableHead>
              <TableHead>Risque</TableHead>
              <TableHead>Statut</TableHead>
              {onExplain && <TableHead className="w-14"></TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginated.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} className="h-24 text-center text-muted-foreground">
                  Aucune transaction correspondante.
                </TableCell>
              </TableRow>
            ) : (
              paginated.map((tx) => (
                <TableRow
                  key={tx.tx_id}
                  className={cn(tx.is_fraud_predicted && "bg-red-50/30 hover:bg-red-50/60")}
                >
                  <TableCell className="font-mono text-xs">{tx.tx_id}</TableCell>
                  <TableCell className="text-xs">{tx.type ?? "—"}</TableCell>
                  <TableCell className="text-right text-xs">
                    {tx.amount != null
                      ? tx.amount.toLocaleString("fr-FR", { minimumFractionDigits: 2 })
                      : "—"}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs">
                    {tx.xgb_score != null ? formatScore(tx.xgb_score) : "—"}
                  </TableCell>
                  <TableCell className="text-right font-mono text-xs">
                    {tx.ae_score != null ? formatScore(tx.ae_score) : "—"}
                  </TableCell>
                  <TableCell>
                    <Badge variant={RISK_BADGE_VARIANT[tx.risk_level]}>
                      {RISK_LABELS_FR[tx.risk_level]}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 text-xs font-medium",
                        tx.is_fraud_predicted ? "text-red-600" : "text-green-600"
                      )}
                    >
                      <span
                        className={cn(
                          "h-1.5 w-1.5 rounded-full",
                          tx.is_fraud_predicted ? "bg-red-500" : "bg-green-500"
                        )}
                      />
                      {tx.is_fraud_predicted ? "Fraude" : "Normal"}
                    </span>
                  </TableCell>
                  {onExplain && (
                    <TableCell>
                      {tx.is_fraud_predicted && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7"
                          onClick={() => onExplain(tx.tx_id)}
                          title="Expliquer"
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </TableCell>
                  )}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {filtered.length.toLocaleString("fr-FR")} résultats
          {filtered.length !== transactions.length && ` (sur ${transactions.length.toLocaleString("fr-FR")})`}
        </span>
        <div className="flex items-center gap-2">
          <Select
            value={pageSize.toString()}
            onValueChange={(v) => { setPageSize(parseInt(v) as typeof pageSize); setPage(0); }}
          >
            <SelectTrigger className="h-7 w-[70px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZES.map((s) => (
                <SelectItem key={s} value={s.toString()}>{s} / page</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon"
            className="h-7 w-7"
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="min-w-[60px] text-center">
            {page + 1} / {totalPages || 1}
          </span>
          <Button
            variant="outline"
            size="icon"
            className="h-7 w-7"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => p + 1)}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
