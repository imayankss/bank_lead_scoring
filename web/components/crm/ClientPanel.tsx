"use client";

import { useEffect } from "react";

import { FLAG_LABEL, PRODUCT_LABEL, compact, nextStep, pct, reasons, type Client } from "@/lib/crm";

export function ClientPanel({ client: c, onClose }: { client: Client; onClose: () => void }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);

  const top = Math.max(...c.p);
  const peak = Math.max(...c.spend, 1);

  return (
    <aside className="drawer" role="dialog" aria-label={`Client ${c.id}`}>
      <div className="drawer-head">
        <div>
          <span className="label">Client · rank {c.rank}</span>
          <h3 className="mono">{c.id}</h3>
        </div>
        <button type="button" className="toggle" onClick={onClose} aria-label="Close client panel">Close</button>
      </div>

      <div className="callout">{nextStep(c)}</div>

      <div className="cells">
        <div className="cell"><span className="label">Lead score</span><span className={`value tier-${c.tier}`}>{c.score} · {c.tier}</span><span className="note">rank by propensity × value</span></div>
        <div className="cell"><span className="label">Any product</span><span className="value model">{pct(c.p_any)}</span><span className="note">chance in the next 3 months</span></div>
      </div>

      <h4 className="label">Chance of each product, next 3 months</h4>
      <ul className="bars">
        {c.p.map((v, i) => (
          <li key={i} className={i + 1 === c.next ? "best" : undefined}>
            <span>{PRODUCT_LABEL[i + 1]}{i + 1 === c.next ? " · next best" : ""}</span>
            <span className="bar"><i style={{ width: `${(v / top) * 100}%` }} /></span>
            <span className="mono">{pct(v)}</span>
          </li>
        ))}
      </ul>

      <h4 className="label">Why this client</h4>
      <ul className="reasons">
        {reasons(c).map((r) => (
          <li key={r.text}><span className={r.up ? "up" : "down"} aria-label={r.up ? "raises the score" : "lowers the score"}>{r.up ? "▲" : "▼"}</span>{r.text}</li>
        ))}
      </ul>

      <h4 className="label">Transaction volume, last 6 months</h4>
      <div className="spark" aria-label="Volume per 30-day window, oldest first">
        {c.spend.map((v, i) => <i key={i} style={{ height: `${Math.max((v / peak) * 100, 2)}%` }} title={compact(v)} />)}
      </div>

      <dl className="facts">
        <div><dt>Volume, 90 days</dt><dd>{compact(c.vol90)}</dd></div>
        <div><dt>Predicted value, next 3 months</dt><dd>{compact(c.value)} <span className="hint">model estimate</span></dd></div>
        <div><dt>Transactions, 90 days</dt><dd>{c.txn90.toLocaleString()}</dd></div>
        <div><dt>Last transaction</dt><dd>{c.recency} days ago</dd></div>
        <div><dt>Products taken before</dt><dd>{c.bought_before.map((n, i) => `${PRODUCT_LABEL[i + 1].replace("Product ", "P")}: ${n}`).join(" · ")}</dd></div>
        <div><dt>Flags (rules)</dt><dd>{c.flags.length ? c.flags.map((f) => FLAG_LABEL[f] ?? f).join(" · ") : "None"}</dd></div>
      </dl>
    </aside>
  );
}
