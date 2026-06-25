"use client";

import { useMemo, useState } from "react";
import { Line, LineChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { currency, percent } from "@/lib/formatters";
import type { ScenarioData } from "@/lib/types";
import { tooltipStyle } from "@/components/charts/ForecastChart";

const axis = { stroke: "#64748b", fontSize: 12 };
const grid = { stroke: "rgba(255,255,255,0.08)" };

export function ScenarioLab({ data }: { data: ScenarioData }) {
  const [uplift, setUplift] = useState(0);
  const adjusted = useMemo(() => {
    const multiplier = 1 + uplift / 100;
    return data.scenarios
      .filter((row) => row.scenario === "Base")
      .map((row) => ({
        month: row.month,
        adjusted_value: (row.cumulative_expected_value ?? 0) * multiplier
      }));
  }, [data.scenarios, uplift]);

  const finalValue = adjusted[adjusted.length - 1]?.adjusted_value ?? data.base_expected_value;

  return (
    <div className="grid gap-5 xl:grid-cols-[0.65fr_1.35fr]">
      <div className="glass rounded-lg p-5">
        <h3 className="text-base font-semibold text-white">Scenario Controls</h3>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          Adjust conversion lift to simulate expected portfolio value movement.
        </p>
        <label className="mt-6 block text-sm text-slate-300">Conversion lift: {percent(uplift)}</label>
        <input
          type="range"
          min="-30"
          max="30"
          step="1"
          value={uplift}
          onChange={(event) => setUplift(Number(event.target.value))}
          className="mt-4 w-full accent-cyan"
        />
        <div className="mt-6 rounded-lg border border-line bg-white/[0.04] p-4">
          <div className="text-sm text-slate-500">Adjusted 12 month value</div>
          <div className="mt-2 text-2xl font-semibold text-white">{currency(finalValue)}</div>
        </div>
      </div>
      <div className="glass rounded-lg p-5">
        <h3 className="mb-4 text-base font-semibold text-white">Adjusted Scenario Curve</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={adjusted}>
              <CartesianGrid {...grid} />
              <XAxis dataKey="month" {...axis} />
              <YAxis {...axis} tickFormatter={currency} />
              <Tooltip formatter={(value) => currency(Number(value))} contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="adjusted_value" stroke="#22d3ee" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
