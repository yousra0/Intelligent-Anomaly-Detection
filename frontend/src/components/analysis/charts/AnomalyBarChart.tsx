"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { TransactionResult } from "@/types";

interface AnomalyBarChartProps {
  transactions: TransactionResult[];
}

const RISK_COLORS: Record<string, string> = {
  CRITIQUE: "#C00000",
  ELEVE: "#D04A02",
  FAIBLE: "#008246",
};

export function AnomalyBarChart({ transactions }: AnomalyBarChartProps) {
  // Group frauds by type
  const typeMap = new Map<string, number>();
  transactions
    .filter((t) => t.is_fraud_predicted)
    .forEach((t) => {
      const type = t.type ?? "Inconnu";
      typeMap.set(type, (typeMap.get(type) ?? 0) + 1);
    });

  const data = Array.from(typeMap.entries())
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count);

  const riskMap = new Map<string, number>();
  transactions.forEach((t) => {
    riskMap.set(t.risk_level, (riskMap.get(t.risk_level) ?? 0) + 1);
  });
  const riskData = Array.from(riskMap.entries()).map(([level, count]) => ({
    level,
    count,
    fill: RISK_COLORS[level] ?? "#888",
  }));

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
        Anomalies par niveau de risque
      </p>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={riskData} margin={{ left: 0, right: 12 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="level" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip
              formatter={(v: number) => [v.toLocaleString("fr-FR"), "Transactions"]}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {riskData.map((entry, idx) => (
                <Cell key={idx} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {data.length > 0 && (
        <>
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mt-4">
            Fraudes par type de transaction
          </p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ left: 0, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip
                  formatter={(v: number) => [v.toLocaleString("fr-FR"), "Fraudes"]}
                />
                <Bar dataKey="count" fill="#D04A02" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
