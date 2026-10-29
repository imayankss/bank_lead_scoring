"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { axis, grid, tooltip } from "@/components/results/chart";

export type Ranker = {
  model_id: string;
  role: string;
  auc: number;
  auc_ci_lower: number;
  auc_ci_upper: number;
  lift_at_10pct: number;
  lift_at_10pct_ci_lower: number;
  lift_at_10pct_ci_upper: number;
};

const METRICS = {
  lift: { label: "Lift in top 10%", key: "lift_at_10pct", chance: 1, fmt: (v: number) => `${v.toFixed(2)}×` },
  auc: { label: "ROC AUC", key: "auc", chance: 0.5, fmt: (v: number) => v.toFixed(3) }
} as const;

export function RankerChart({ rankers, labels }: { rankers: Ranker[]; labels: Record<string, string> }) {
  const [metric, setMetric] = useState<keyof typeof METRICS>("lift");
  const m = METRICS[metric];
  const data = [...rankers]
    .sort((a, b) => b[m.key] - a[m.key])
    .map((r) => {
      const v = r[m.key];
      // ErrorBar takes [minus, plus] distances from the bar end.
      return { name: labels[r.model_id] ?? r.model_id, v, ci: [v - r[`${m.key}_ci_lower`], r[`${m.key}_ci_upper`] - v], id: r.model_id, baseline: r.role === "baseline" };
    });

  return (
    <div className="panel">
      <div className="controls">
        <span className="label">Metric</span>
        <div className="chips" role="group" aria-label="Choose a metric">
          {(Object.keys(METRICS) as (keyof typeof METRICS)[]).map((k) => (
            <button key={k} type="button" className="chip" aria-pressed={k === metric} onClick={() => setMetric(k)}>
              {METRICS[k].label}
            </button>
          ))}
        </div>
      </div>
      <div className="chart tall">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid {...grid} horizontal={false} />
            <XAxis type="number" {...axis} domain={metric === "auc" ? [0.45, 0.8] : [0, 4.5]} ticks={metric === "auc" ? [0.5, 0.6, 0.7, 0.8] : [0, 1, 2, 3, 4]} tickFormatter={m.fmt} />
            <YAxis type="category" dataKey="name" {...axis} width={150} />
            <Tooltip {...tooltip} cursor={{ fill: "var(--surface-3)" }} formatter={(v) => [m.fmt(Number(v)), m.label]} />
            <ReferenceLine x={m.chance} stroke="var(--axis)" strokeDasharray="3 3" />
            <Bar dataKey="v" barSize={14} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.id} fill={d.id === "lightgbm" ? "var(--series-1)" : d.baseline ? "var(--series-2)" : "var(--text-muted)"} />
              ))}
              <ErrorBar dataKey="ci" direction="x" stroke="var(--text-primary)" width={6} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--series-1)" }} />production model</span>
        <span><i style={{ background: "var(--text-muted)" }} />lead score (propensity × value)</span>
        <span><i style={{ background: "var(--series-2)" }} />baselines</span>
      </div>
      <p className="hint">Oct 2022 test month, 100,224 clients, scored once — whiskers are 95% bootstrap intervals; dashed line is chance</p>
    </div>
  );
}
