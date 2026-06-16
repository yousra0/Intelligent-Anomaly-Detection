"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  TrendingUp,
  ClipboardList,
  Brain,
  BarChart2,
  Clock,
  Hash,
} from "lucide-react";
import { cn, formatScore, formatDateTime, RISK_LABELS_FR } from "@/lib/utils";
import type { ExplainResponse, RiskLevel } from "@/types";

interface ExplanationCardProps {
  explanation: ExplainResponse;
}

const RISK_CONFIG: Record<
  RiskLevel,
  { variant: "critique" | "eleve" | "faible"; icon: typeof AlertTriangle; color: string }
> = {
  CRITIQUE: { variant: "critique", icon: AlertTriangle, color: "text-red-600" },
  ELEVE:    { variant: "eleve",    icon: AlertTriangle, color: "text-orange-600" },
  FAIBLE:   { variant: "faible",   icon: CheckCircle2,  color: "text-green-600" },
};

const FEATURE_FR: Record<string, string> = {
  balance_diff_orig: "Différence solde émetteur",
  log_amount: "Montant (log)",
  dest_zero_balance: "Destination solde nul",
  is_transfer_or_cashout: "TRANSFER / CASH_OUT",
  high_risk_hour: "Heure à risque",
  type_CASH_OUT: "Type CASH_OUT",
  type_TRANSFER: "Type TRANSFER",
  type_PAYMENT: "Type PAYMENT",
  type_CASH_IN: "Type CASH_IN",
  type_DEBIT: "Type DEBIT",
  step: "Étape temporelle",
  hour: "Heure",
  day: "Jour",
  week: "Semaine",
};

function featureFR(name: string): string {
  return FEATURE_FR[name] ?? name;
}

export function ExplanationCard({ explanation }: ExplanationCardProps) {
  const { llm } = explanation;
  const riskCfg = RISK_CONFIG[explanation.risk_level];
  const RiskIcon = riskCfg.icon;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header card */}
      <Card className={cn("border-l-4", {
        "border-l-red-500": explanation.risk_level === "CRITIQUE",
        "border-l-orange-500": explanation.risk_level === "ELEVE",
        "border-l-green-500": explanation.risk_level === "FAIBLE",
      })}>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <RiskIcon className={cn("h-5 w-5", riskCfg.color)} />
                <span className="text-lg font-bold">Transaction #{explanation.tx_id}</span>
              </div>
              <Badge variant={riskCfg.variant} className="text-xs">
                Risque {RISK_LABELS_FR[explanation.risk_level]}
              </Badge>
            </div>
            <div className="text-right text-sm space-y-1">
              {explanation.xgb_score != null && (
                <p className="text-muted-foreground">
                  XGBoost: <span className="font-mono font-semibold">{formatScore(explanation.xgb_score)}</span>
                </p>
              )}
              {explanation.ae_score != null && (
                <p className="text-muted-foreground">
                  AutoEncoder: <span className="font-mono font-semibold">{formatScore(explanation.ae_score)}</span>
                </p>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {/* LLM Explanation */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Brain className="h-4 w-4 text-pwc-orange" />
              Analyse LLM (Groq)
              {llm.status === "fallback" && (
                <span className="text-xs text-muted-foreground">(mode règles)</span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {llm.resume && (
              <p className="rounded-lg bg-muted/50 p-3 text-sm leading-relaxed italic">
                "{llm.resume}"
              </p>
            )}

            {llm.raisons && llm.raisons.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Facteurs de risque identifiés
                </p>
                <ul className="space-y-1.5">
                  {llm.raisons.map((r, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-red-100 text-[10px] font-bold text-red-700">
                        {i + 1}
                      </span>
                      <span className="text-xs leading-relaxed">{r}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {llm.actions_recommandees && llm.actions_recommandees.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Actions recommandées
                </p>
                <ul className="space-y-1.5">
                  {llm.actions_recommandees.map((a, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <ClipboardList className="mt-0.5 h-3.5 w-3.5 shrink-0 text-pwc-orange" />
                      <span className="text-xs leading-relaxed">{a}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {llm._audit && (
              <>
                <Separator />
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div className="flex items-center gap-1.5">
                    <Clock className="h-3 w-3" />
                    <span>{formatDateTime(llm._audit.timestamp_utc)}</span>
                  </div>
                  <div className="flex items-start gap-1.5">
                    <Hash className="h-3 w-3 mt-0.5 shrink-0" />
                    <span className="font-mono text-[10px] break-all">{llm._audit.hash}</span>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* SHAP + AE features */}
        <div className="space-y-4">
          {/* SHAP XGBoost */}
          {explanation.shap_values_xgb && Object.keys(explanation.shap_values_xgb).length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <TrendingUp className="h-4 w-4 text-blue-500" />
                  SHAP — XGBoost
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {Object.entries(explanation.shap_values_xgb)
                  .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                  .slice(0, 6)
                  .map(([feat, val]) => (
                    <div key={feat} className="flex items-center gap-2 text-xs">
                      <span className="w-40 truncate text-muted-foreground" title={featureFR(feat)}>
                        {featureFR(feat)}
                      </span>
                      <div className="flex flex-1 items-center gap-1">
                        <div
                          className={cn(
                            "h-2 rounded-full",
                            val > 0 ? "bg-red-400" : "bg-green-400"
                          )}
                          style={{ width: `${Math.min(Math.abs(val) * 80, 100)}%` }}
                        />
                      </div>
                      <span className={cn("w-16 text-right font-mono", val > 0 ? "text-red-600" : "text-green-600")}>
                        {val > 0 ? "+" : ""}{val.toFixed(4)}
                      </span>
                    </div>
                  ))}
              </CardContent>
            </Card>
          )}

          {/* AE feature errors */}
          {explanation.ae_top_features && explanation.ae_top_features.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <BarChart2 className="h-4 w-4 text-purple-500" />
                  AutoEncoder — Erreurs par feature
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {explanation.ae_top_features.map((f) => (
                  <div key={f.feature} className="flex items-center gap-2 text-xs">
                    <span className="w-40 truncate text-muted-foreground" title={featureFR(f.feature)}>
                      {featureFR(f.feature)}
                    </span>
                    <div className="flex flex-1 items-center gap-1">
                      <div
                        className="h-2 rounded-full bg-purple-400"
                        style={{ width: `${Math.min((f.error / (explanation.ae_top_features![0].error || 1)) * 100, 100)}%` }}
                      />
                    </div>
                    <span className="w-16 text-right font-mono text-purple-700">
                      {f.error.toFixed(2)}
                    </span>
                  </div>
                ))}
                <p className="text-[10px] text-muted-foreground mt-2">
                  Erreur = |x − AE(x)| · Plus l'erreur est élevée, plus la feature est anormale.
                </p>
              </CardContent>
            </Card>
          )}

          {/* LIME rules */}
          {explanation.lime_rules && explanation.lime_rules.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Info className="h-4 w-4 text-cyan-500" />
                  LIME — Règles interprétables
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-1.5">
                  {explanation.lime_rules.map((rule, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs">
                      <span className="font-mono text-cyan-700 bg-cyan-50 px-1.5 py-0.5 rounded text-[10px]">
                        {i + 1}
                      </span>
                      <span>{rule}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
