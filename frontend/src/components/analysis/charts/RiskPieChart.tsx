"use client";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { PredictResponse } from "@/types";

interface RiskPieChartProps {
  result: PredictResponse;
}

export function RiskPieChart({ result }: RiskPieChartProps) {
  const normal = result.n_transactions - result.n_fraud;
  const data = [
    { name: "Normal", value: normal, color: "#008246" },
    { name: "Anomalie", value: result.n_fraud, color: "#C00000" },
  ];

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={90}
            paddingAngle={3}
            dataKey="value"
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value: number, name: string) => [
              value.toLocaleString("fr-FR"),
              name,
            ]}
          />
          <Legend
            formatter={(value) => (
              <span className="text-xs text-gray-600">{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
