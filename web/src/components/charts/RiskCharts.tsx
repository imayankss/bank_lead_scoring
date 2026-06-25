"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { tooltipStyle } from "./ForecastChart";

const axis = { stroke: "#64748b", fontSize: 12 };
const grid = { stroke: "rgba(255,255,255,0.08)" };

export function RiskDistributionChart({ data }: { data: Array<Record<string, number | string>> }) {
  const counts = data.reduce<Record<string, number>>((acc, row) => {
    const category = String(row.risk_category ?? "Unknown");
    acc[category] = (acc[category] ?? 0) + 1;
    return acc;
  }, {});
  const chartData = Object.entries(counts).map(([category, count]) => ({ category, count }));
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="category" {...axis} />
          <YAxis {...axis} />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar dataKey="count" fill="#fb7185" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
