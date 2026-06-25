"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { currency } from "@/lib/formatters";
import { tooltipStyle } from "./ForecastChart";

const axis = { stroke: "#64748b", fontSize: 12 };
const grid = { stroke: "rgba(255,255,255,0.08)" };

export function ResidualChart({ data }: { data: Array<Record<string, number | string>> }) {
  const chartData = data.slice(0, 60).map((row, index) => ({
    index: index + 1,
    residual: Number(row.residual ?? 0)
  }));
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="index" {...axis} />
          <YAxis {...axis} tickFormatter={currency} />
          <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
          <Line type="monotone" dataKey="residual" stroke="#fb7185" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ErrorBucketChart({ data }: { data: Array<Record<string, number | string>> }) {
  const buckets = data.reduce<Record<string, number>>((acc, row) => {
    const bucket = String(row.error_bucket ?? "unknown");
    acc[bucket] = (acc[bucket] ?? 0) + 1;
    return acc;
  }, {});
  const chartData = Object.entries(buckets).map(([bucket, count]) => ({ bucket, count }));
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="bucket" {...axis} />
          <YAxis {...axis} />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar dataKey="count" fill="#f59e0b" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
