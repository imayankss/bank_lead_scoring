"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { decimal } from "@/lib/formatters";
import type { ModelRecord } from "@/lib/types";
import { tooltipStyle } from "./ForecastChart";

const axis = { stroke: "#64748b", fontSize: 12 };
const grid = { stroke: "rgba(255,255,255,0.08)" };

export function ModelMetricChart({ data }: { data: ModelRecord[] }) {
  const chartData = data.map((row) => ({
    model: row.model_name ?? row.display_name ?? row.model_id,
    business_score: row.business_score ?? 0,
    rmse: row.rmse ?? 0
  }));

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ left: 40 }}>
          <CartesianGrid {...grid} />
          <XAxis type="number" {...axis} tickFormatter={(value) => decimal(Number(value), 2)} />
          <YAxis type="category" dataKey="model" {...axis} width={140} />
          <Tooltip formatter={(value) => decimal(Number(value), 3)} contentStyle={tooltipStyle} />
          <Bar dataKey="business_score" fill="#34d399" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
