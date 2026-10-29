import Link from "next/link";

import { LeadDesk } from "@/components/crm/LeadDesk";
import { ThemeToggle } from "@/components/results/ThemeToggle";
import { compact, type Crm } from "@/lib/crm";
import { readJson } from "@/lib/data";

const REPO = "https://github.com/imayankss/bank_lead_scoring";

export default async function LeadDeskPage() {
  const crm = await readJson<Crm | null>("crm.json", null);
  if (!crm) {
    return <main className="page"><p className="hint">CRM file missing. Run <code>python -m lead_scoring.web_export</code>.</p></main>;
  }
  const s = crm.summary;
  const hits = Object.fromEntries(crm.test.filter((t) => t.measure === "top1_hit_rate_among_buyers").map((t) => [t.model, t.value]));
  const date = new Date(crm.snapshot).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });

  return (
    <main className="page desk">
      <header>
        <div className="masthead">
          <div>
            <h1 className="title">Lead Desk</h1>
            <p className="subtitle mono">Book as of {date} · next 3 months · {s.clients.toLocaleString()} business clients</p>
          </div>
          <ThemeToggle />
        </div>
        <p className="claim">
          Who to call, what to offer and why. Ranked by chance of taking a product × predicted value, with the next best product for every client.{" "}
          <Link href="/results">How good are these scores? →</Link>
        </p>
      </header>

      <div className="cells">
        <div className="cell"><span className="label">Hot leads</span><span className="value tier-Hot">{(s.tiers.Hot ?? 0).toLocaleString()}</span><span className="note">top 30% by lead score</span></div>
        <div className="cell"><span className="label">Expected takers</span><span className="value model">{Math.round(s.expected_takers).toLocaleString()}</span><span className="note">clients likely to take any product, whole book</span></div>
        <div className="cell"><span className="label">Value at stake</span><span className="value">{compact(s.expected_value)}</span><span className="note">sum of propensity × predicted value (MBD units)</span></div>
        <div className="cell"><span className="label">Right product first</span><span className="value model">{pct0(hits.lightgbm)}</span><span className="note">of past buyers got their product as #1, vs {pct0(hits.most_popular_product)} offering the most popular</span></div>
      </div>

      <h2 className="desk-h">Demand by product <span className="hint">click to filter the list</span></h2>
      <LeadDesk clients={crm.clients} products={crm.products} />

      <p className="hint">
        Showing the top {s.shown.toLocaleString()} clients by lead score. Clients are anonymised business clients from Sber&apos;s public MBD dataset (CC BY 4.0), not real bank customers; products are unnamed in the source.
        Probabilities come from LightGBM models chosen on Jul 2022 and checked once on Oct 2022. Flags and the suggested step are rules.
      </p>

      <footer className="footer">
        Bank Lead Scoring · <Link href="/results">model results</Link> · <a href={REPO}>source</a>
      </footer>
    </main>
  );
}

const pct0 = (v: number | undefined) => (v === undefined ? "—" : `${Math.round(v * 100)}%`);
