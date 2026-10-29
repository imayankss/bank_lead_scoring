"use client";

import { useMemo, useState } from "react";

import { ClientPanel } from "@/components/crm/ClientPanel";
import { FLAG_LABEL, PRODUCT_LABEL, compact, pct, type Client, type Crm } from "@/lib/crm";

const SORTS = {
  score: { label: "Lead score", by: (c: Client) => c.ev },
  p_any: { label: "Propensity", by: (c: Client) => c.p_any },
  value: { label: "Predicted value", by: (c: Client) => c.value },
  vol90: { label: "90-day volume", by: (c: Client) => c.vol90 }
} as const;
const PAGE = 50;

export function LeadDesk({ clients, products }: { clients: Client[]; products: Crm["products"] }) {
  const [product, setProduct] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<keyof typeof SORTS>("score");
  const [shown, setShown] = useState(PAGE);
  const [open, setOpen] = useState<Client | null>(null);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return clients
      .filter((c) => (product === null || c.next === product) && (!q || c.id.startsWith(q)))
      .sort((a, b) => SORTS[sort].by(b) - SORTS[sort].by(a));
  }, [clients, product, query, sort]);
  const listed = (k: number) => clients.filter((c) => c.next === k).length;

  return (
    <>
      <div className="cells products">
        {products.map((p) => (
          <button key={p.product} type="button" className="cell" aria-pressed={product === p.product} onClick={() => { setProduct(product === p.product ? null : p.product); setShown(PAGE); }}>
            <span className="label">{PRODUCT_LABEL[p.product]}</span>
            <span className="value model">{Math.round(p.expected_takers).toLocaleString()}</span>
            <span className="note">expected takers in the book · next best for {p.next_for.toLocaleString()} clients, {listed(p.product)} on this list</span>
          </button>
        ))}
      </div>

      <div className="controls desk-controls">
        <label className="field">
          <span className="label">Sort</span>
          <select value={sort} onChange={(e) => setSort(e.target.value as keyof typeof SORTS)}>
            {Object.entries(SORTS).map(([k, s]) => <option key={k} value={k}>{s.label}</option>)}
          </select>
        </label>
        <label className="field">
          <span className="label">Client</span>
          <input type="search" placeholder="ID starts with…" value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        <span className="hint">{rows.length.toLocaleString()} clients{product ? ` · next best ${PRODUCT_LABEL[product]}` : ""}</span>
      </div>

      <div className="table-wrap desk-table">
        <table>
          <thead>
            <tr>
              <th className="num">#</th><th>Client</th><th>Tier</th><th className="num">Score</th><th className="num">Any product</th>
              <th>Next best product</th><th className="num">Pred. value</th><th className="num">90-day volume</th><th className="num">Last txn</th><th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, shown).map((c) => (
              <tr key={c.id} className="row" tabIndex={0} onClick={() => setOpen(c)} onKeyDown={(e) => e.key === "Enter" && setOpen(c)} aria-selected={open?.id === c.id}>
                <td className="num">{c.rank}</td>
                <td className="mono">{c.id}</td>
                <td><span className={`tier tier-${c.tier}`}>{c.tier}</span></td>
                <td className="num">{c.score}</td>
                <td className="num">{pct(c.p_any)}</td>
                <td>{PRODUCT_LABEL[c.next]} <span className="hint">{pct(c.p[c.next - 1])}</span></td>
                <td className="num">{compact(c.value)}</td>
                <td className="num">{compact(c.vol90)}</td>
                <td className="num">{c.recency}d</td>
                <td className="flags">{c.flags.map((f) => FLAG_LABEL[f] ?? f).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown < rows.length && (
        <button type="button" className="chip more" onClick={() => setShown(shown + PAGE)}>Show {Math.min(PAGE, rows.length - shown)} more</button>
      )}

      {open && <ClientPanel client={open} onClose={() => setOpen(null)} />}
    </>
  );
}
