"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { currency } from "@/lib/formatters";
import type { ForecastPoint } from "@/lib/types";

const axis = { stroke: "#64748b", fontSize: 12 };
const grid = { stroke: "rgba(255,255,255,0.08)" };

export function ActualVsPredictedChart({ data }: { data: ForecastPoint[] }) {
  const chartData = data.slice(0, 40).map((row, index) => ({
    name: row.customer_id ?? `${index + 1}`,
    actual: row.actual_future_revenue_12m ?? 0,
    predicted: row.predicted_cltv ?? 0
  }));

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="name" {...axis} hide />
          <YAxis {...axis} tickFormatter={currency} />
          <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
          <Legend />
          <Line type="monotone" dataKey="actual" stroke="#34d399" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="predicted" stroke="#22d3ee" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ScenarioForecastChart({ data }: { data: ForecastPoint[] }) {
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="month" {...axis} />
          <YAxis {...axis} tickFormatter={currency} />
          <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
          <Legend />
          <Line type="monotone" dataKey="cumulative_expected_value" stroke="#22d3ee" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MultiScenarioChart({ data }: { data: ForecastPoint[] }) {
  const byMonth = Array.from({ length: 12 }, (_, index) => {
    const month = index + 1;
    const row: Record<string, number> = { month };
    data.filter((item) => item.month === month).forEach((item) => {
      row[item.scenario ?? "Scenario"] = item.cumulative_expected_value ?? 0;
    });
    return row;
  });

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={byMonth}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="month" {...axis} />
          <YAxis {...axis} tickFormatter={currency} />
          <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
          <Legend />
          <Area type="monotone" dataKey="Optimistic" stroke="#34d399" fill="#34d399" fillOpacity={0.16} />
          <Area type="monotone" dataKey="Base" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.12} />
          <Area type="monotone" dataKey="Pessimistic" stroke="#fb7185" fill="#fb7185" fillOpacity={0.10} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function LeadCategoryChart({ data }: { data: Record<string, number> }) {
  const chartData = Object.entries(data).map(([category, count]) => ({ category, count }));
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="category" {...axis} />
          <YAxis {...axis} />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar dataKey="count" fill="#22d3ee" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export const tooltipStyle = {
  background: "rgba(7, 9, 13, 0.95)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: "8px",
  color: "#f8fafc"
};
