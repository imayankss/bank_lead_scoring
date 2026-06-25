"use client";

import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { currency } from "@/lib/formatters";
import { tooltipStyle } from "./ForecastChart";

const axis = { stroke: "#64748b", fontSize: 12 };
const grid = { stroke: "rgba(255,255,255,0.08)" };

export function TrendChart({ data }: { data: Array<Record<string, number | string>> }) {
  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data.slice(-48)}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="period" {...axis} hide />
          <YAxis {...axis} tickFormatter={currency} />
          <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
          <Line type="monotone" dataKey="total_amount" stroke="#22d3ee" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="rolling_3m_amount" stroke="#34d399" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MonthPatternChart({ data }: { data: Array<Record<string, number>> }) {
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="month" {...axis} />
          <YAxis {...axis} tickFormatter={currency} />
          <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
          <Bar dataKey="avg_total_amount" fill="#f59e0b" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
