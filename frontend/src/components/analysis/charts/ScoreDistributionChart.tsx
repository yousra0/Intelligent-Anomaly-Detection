"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { TransactionResult } from "@/types";

interface ScoreDistributionChartProps {
  transactions: TransactionResult[];
  threshold?: number;
}

export function ScoreDistributionChart({
  transactions,
  threshold = 0.355,
}: ScoreDistributionChartProps) {
  // Build histogram of xgb_scores in 20 buckets
  const scores = transactions
    .map((t) => t.xgb_score ?? t.ae_score ?? 0)
    .filter((s) => s !== undefined);

  const buckets = 20;
  const hist = new Array(buckets).fill(0);
  scores.forEach((s) => {
    const idx = Math.min(Math.floor(s * buckets), buckets - 1);
    hist[idx]++;
  });

  const data = hist.map((count, i) => ({
    score: ((i + 0.5) / buckets).toFixed(2),
    count,
    fraud: i / buckets >= threshold ? count : 0,
    normal: i / buckets < threshold ? count : 0,
  }));

  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ left: 0, right: 12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="score"
            tick={{ fontSize: 10 }}
            label={{ value: "Score", position: "insideBottom", offset: -2, fontSize: 11 }}
          />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(v: number, name: string) => [
              v.toLocaleString("fr-FR"),
              name === "normal" ? "Normal" : "Fraude potentielle",
            ]}
            labelFormatter={(l) => `Score ~${l}`}
          />
          <ReferenceLine
            x={(threshold).toFixed(2)}
            stroke="#D04A02"
            strokeDasharray="4 2"
            label={{ value: "Seuil", fontSize: 10, fill: "#D04A02" }}
          />
          <Area
            type="monotone"
            dataKey="normal"
            stackId="1"
            stroke="#008246"
            fill="#008246"
            fillOpacity={0.3}
          />
          <Area
            type="monotone"
            dataKey="fraud"
            stackId="1"
            stroke="#C00000"
            fill="#C00000"
            fillOpacity={0.4}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
