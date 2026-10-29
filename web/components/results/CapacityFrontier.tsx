"use client";

import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { axis, grid, tooltip } from "@/components/results/chart";
import { readout, type GainsRow } from "@/lib/frontier";

const TIERS = [1, 2, 3, 5, 10];
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

export function CapacityFrontier({ gains }: { gains: GainsRow[] }) {
  const [k, setK] = useState(1);
  const r = readout(gains, k);
  const totalBuyers = gains.reduce((s, g) => s + g.converters, 0);
  const curve = [{ x: 0, model: 0, random: 0 }, ...gains.map((g, i) => ({ x: (i + 1) * 10, model: g.cumulative_share_of_converters * 100, random: (i + 1) * 10 }))];
  const show = (v: number | undefined, f: (v: number) => string) => (v === undefined ? "—" : f(v));

  return (
    <div className="panel">
      <div className="controls">
        <span className="label">Call capacity</span>
        <div className="slider">
          <input type="range" min={1} max={10} step={1} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="Share of clients called" />
          <span className="mono">top {k * 10}% of clients</span>
        </div>
      </div>
      <div className="controls">
        <span className="label">Measured tiers</span>
        <div className="chips" role="group" aria-label="Measured tiers">
          {TIERS.map((t) => (
            <button key={t} type="button" className="chip" aria-pressed={t === k} onClick={() => setK(t)}>
              {t * 10}%
            </button>
          ))}
        </div>
      </div>
      <div className="chart">
        <ResponsiveContainer>
          <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid {...grid} vertical={false} />
            <XAxis dataKey="x" type="number" domain={[0, 100]} ticks={[0, 20, 40, 60, 80, 100]} unit="%" {...axis} />
            <YAxis domain={[0, 100]} unit="%" width={44} {...axis} />
            <Tooltip {...tooltip} formatter={(v, name) => [`${Number(v).toFixed(1)}%`, name === "model" ? "lead-score ranking" : "random calling"]} labelFormatter={(x) => `top ${x}% called`} />
            <ReferenceLine x={k * 10} stroke="var(--series-1)" strokeDasharray="3 3" />
            <Line dataKey="random" stroke="var(--series-2)" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
            <Line dataKey="model" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--series-1)" }} />share of product takers reached, ranked by the lead score</span>
        <span><i style={{ background: "var(--series-2)" }} />random calling</span>
      </div>
      <div className="cells">
        <div className="cell">
          <span className="label">Takers reached</span>
          <span className="value model">{show(r?.share, (v) => pct(v))}</span>
          <span className="note">{show(r?.buyers, (v) => v.toLocaleString())} of {totalBuyers.toLocaleString()} clients who took a product</span>
        </div>
        <div className="cell">
          <span className="label">Clients called</span>
          <span className="value">{show(r?.clients, (v) => v.toLocaleString())}</span>
          <span className="note">the capacity you chose</span>
        </div>
        <div className="cell">
          <span className="label">Hit rate</span>
          <span className="value">{show(r?.hitRate, (v) => pct(v, 2))}</span>
          <span className="note">share of calls that reach a taker</span>
        </div>
        <div className="cell">
          <span className="label">Lift vs random</span>
          <span className="value model">{show(r?.lift, (v) => `${v.toFixed(2)}×`)}</span>
          <span className="note">hit rate over the base rate</span>
        </div>
        <div className="cell">
          <span className="label">Takers missed</span>
          <span className="value miss">{show(r?.buyers, (v) => (totalBuyers - v).toLocaleString())}</span>
          <span className="note">never called</span>
        </div>
      </div>
      <p className="hint">measured deciles of the Oct 2022 test month only — no interpolation, nothing recomputed</p>
    </div>
  );
}
