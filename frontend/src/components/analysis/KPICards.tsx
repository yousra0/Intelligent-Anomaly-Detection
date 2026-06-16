"use client";

import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, AlertTriangle, DollarSign, Database, ShieldCheck } from "lucide-react";
import { formatAmount, formatPercent } from "@/lib/utils";
import type { PredictResponse } from "@/types";

interface KPICardsProps {
  result: PredictResponse;
}

export function KPICards({ result }: KPICardsProps) {
  const kpis = [
    {
      label: "Total transactions",
      value: result.n_transactions.toLocaleString("fr-FR"),
      icon: Database,
      color: "text-blue-600",
      bg: "bg-blue-50",
    },
    {
      label: "Anomalies détectées",
      value: result.n_fraud.toLocaleString("fr-FR"),
      icon: AlertTriangle,
      color: "text-red-600",
      bg: "bg-red-50",
    },
    {
      label: "Taux de fraude",
      value: formatPercent(result.fraud_rate_pct),
      icon: TrendingUp,
      color: "text-orange-600",
      bg: "bg-orange-50",
    },
    {
      label: "Montant à risque",
      value: formatAmount(result.amount_at_risk),
      icon: DollarSign,
      color: "text-purple-600",
      bg: "bg-purple-50",
    },
    {
      label: "Modèle utilisé",
      value: result.model_used,
      icon: ShieldCheck,
      color: "text-green-600",
      bg: "bg-green-50",
      small: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      {kpis.map((kpi) => {
        const Icon = kpi.icon;
        return (
          <Card key={kpi.label} className="animate-fade-in">
            <CardContent className="p-4">
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-muted-foreground">{kpi.label}</p>
                  <p
                    className={`mt-1 font-bold ${kpi.small ? "text-sm" : "text-xl"} truncate`}
                    title={kpi.value}
                  >
                    {kpi.value}
                  </p>
                </div>
                <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${kpi.bg}`}>
                  <Icon className={`h-4 w-4 ${kpi.color}`} />
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
