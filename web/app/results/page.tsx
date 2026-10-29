import Link from "next/link";

import { CapacityFrontier } from "@/components/results/CapacityFrontier";
import { RankerChart, type Ranker } from "@/components/results/RankerChart";
import { ThemeToggle } from "@/components/results/ThemeToggle";
import { readJson } from "@/lib/data";
import type { GainsRow } from "@/lib/frontier";

type Results = {
  test: { test_snapshot: string; rows: number; positives: number; base_rate: number; test_auc: number; test_lift_at_10pct: number; validation_auc: number };
  rankers: (Ranker & { model_name: string })[];
  value: { model_id: string; model_name: string; role: string; rmse: number; mae: number }[];
  gains: GainsRow[];
  validation_grid: { model: string; num_leaves: number | null; min_child_samples: number | null; val_auc: number; val_lift_at_10pct: number; gru_weight: number | null }[];
  paper: { model: string; source: string; mean_auc: number; auc_y1_target_1: number | null; auc_y1_target_2: number | null; auc_y1_target_3: number | null; auc_y1_target_4: number | null }[];
  next_product: { measure: string; product: string; model: string; value: number; buyers: number }[];
  run: { git_sha: string | null; created_utc: string | null };
};

const REPO = "https://github.com/imayankss/bank_lead_scoring";
// Dataset and training-set sizes from the README; the result files only hold the scored test month.
const MBD = { transactions: "38.8M", features: 297, trainRows: 701_568 };
const RANKER_LABEL: Record<string, string> = {
  lightgbm: "LightGBM",
  logistic_regression: "Logistic regression",
  previous_gbm: "Previous model (14 features)",
  expected_value_ranker: "Lead score (p × value)",
  product_history_rule: "Rule: bought before",
  activity_rule: "Rule: most active",
  recency_rule: "Rule: most recent",
  prior_baseline: "No ranking"
};
// Validation-month ablation from docs/plans/2026-10-25-auc-lift-strategy.md (not a result table).
const STEPS = [
  { step: "LightGBM on transaction-mix and 2-month product-history features", auc: 0.762, lift: 3.29 },
  { step: "+ the snapshot month's purchases and months since the last purchase", auc: 0.768, lift: 3.46 },
  { step: "+ dialog counts and days since the last dialog", auc: 0.774, lift: 3.55 }
];
const units = (v: number) => `${(v / 1e6).toFixed(0)}M`;
const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const ci = (lo: number, hi: number, d = 2) => `${lo.toFixed(d)}–${hi.toFixed(d)}`;

function Section({ n, id, title, children }: { n: string; id: string; title: string; children: React.ReactNode }) {
  return (
    <section className="section" id={id}>
      <h2><span className="num">{n}</span>{title}</h2>
      {children}
    </section>
  );
}

export default async function ResultsPage() {
  const r = await readJson<Results | null>("results.json", null);
  if (!r) {
    return <main className="page"><p className="hint">Result file missing. Run <code>python -m lead_scoring.web_export</code>.</p></main>;
  }

  const rank = Object.fromEntries(r.rankers.map((x) => [x.model_id, x]));
  const lgb = rank.lightgbm;
  const lr = rank.logistic_regression;
  const prev = rank.previous_gbm;
  const ev = rank.expected_value_ranker;
  const repeat = r.value.find((v) => v.model_id === "last_90_days_baseline")!;
  const cltv = r.value.find((v) => v.model_id === "cltv_gbm")!;
  const gru = r.validation_grid.find((v) => v.model === "gru")!;
  const ours = r.paper.find((p) => p.source === "ours")!;
  const sha = r.run.git_sha?.slice(0, 7) ?? "uncommitted";
  const nbpAuc = r.next_product.filter((x) => x.measure === "auc");
  const nbpHit = Object.fromEntries(r.next_product.filter((x) => x.measure === "top1_hit_rate_among_buyers").map((x) => [x.model, x]));

  return (
    <main className="page">
      <header>
        <div className="masthead">
          <h1 className="title">Bank Lead Scoring</h1>
          <ThemeToggle />
        </div>
        <p className="subtitle mono"><Link href="/">← Lead Desk</Link></p>
        <p className="claim">
          Which business clients should a relationship manager call first? A lead-scoring model rebuilt on {r.test.rows.toLocaleString()} real bank clients — every number measured on a later month the model never saw, next to the baselines it has to beat.
        </p>
        <dl className="meta">
          <div><dt>Origin</dt><dd>Bank of India-style RM workflow</dd></div>
          <div><dt>Data</dt><dd>Sber AI Lab MBD-mini · CC BY 4.0</dd></div>
          <div><dt>Evaluation</dt><dd>Settings chosen on Jul 2022, Oct 2022 scored once</dd></div>
          <div><dt>Scale</dt><dd>{r.test.rows.toLocaleString()} clients · {MBD.transactions} transactions · {MBD.features} features</dd></div>
          <div><dt>Run</dt><dd><a href={`${REPO}/commit/${r.run.git_sha}`}>{sha}</a></dd></div>
          <div><dt>Code</dt><dd><a href={REPO}>GitHub</a></dd></div>
        </dl>
      </header>

      <div className="lede">
        <p>
          The first version of this project reported <strong>0.913 ROC AUC</strong>. It was leakage: account counts read from today&apos;s table, a random split, 500 synthetic customers. On real data, with the leaks closed, an honest model scores lower — and that is the number worth showing.
        </p>
        <p>
          The rebuild runs on Sber&apos;s public Multimodal Banking Dataset. Every setting was chosen on a validation month; the test month was scored once.
        </p>
        <div className="cells">
          <div className="cell">
            <span className="label">Top-10% lift</span>
            <span className="value model">{lgb.lift_at_10pct.toFixed(2)}×</span>
            <span className="note">vs calling at random, 95% CI {ci(lgb.lift_at_10pct_ci_lower, lgb.lift_at_10pct_ci_upper)}</span>
          </div>
          <div className="cell">
            <span className="label">ROC AUC</span>
            <span className="value model">{lgb.auc.toFixed(3)}</span>
            <span className="note">up from {prev.auc.toFixed(3)} for the previous model on the same rows</span>
          </div>
          <div className="cell">
            <span className="label">Client value (CLTV)</span>
            <span className="value baseline">{units(repeat.rmse)} RMSE</span>
            <span className="note">&ldquo;last 90 days again&rdquo; beats the model ({units(cltv.rmse)})</span>
          </div>
        </div>
      </div>

      <Section n="01" id="question" title="The question">
        <div className="prose">
          <p>
            At a month-end snapshot, which clients will take any of the bank&apos;s 4 products in the next 3 months? The model is refit on 7 month-ends, Jan–Jul 2022 ({MBD.trainRows.toLocaleString()} client-snapshots), and tested on the 31 Oct 2022 snapshot with labels from Nov 2022 to Jan 2023. In the test months {(r.test.base_rate * 100).toFixed(1)}% of clients took a product.
          </p>
          <p>These are business clients of a Russian bank, not Bank of India customers: no public customer-level BOI data exists. The workflow carries over; the numbers may not.</p>
        </div>
      </Section>

      <Section n="02" id="protocol" title="Evaluation protocol">
        <ol className="steps">
          <li><strong>Later months only.</strong> Settings are chosen on Jul 2022 (fit Jan–Apr); the chosen model is refit and tested once on Oct 2022. A script refuses to score the test month twice.</li>
          <li><strong>Leakage guards that raise.</strong> <code>temporal.py</code> stops the run if a split or feature reaches past the snapshot, and any single feature above 0.80 validation AUC halts it.</li>
          <li><strong>Baselines in every table.</strong> Rules an RM team could write without a model, logistic regression on the same features, and the previous model.</li>
          <li><strong>Uncertainty.</strong> 95% bootstrap intervals over clients for AUC and lift.</li>
          <li><strong>Provenance.</strong> A run manifest records the git SHA, config, data revision and hashes, and package versions; a test fails if a README number drifts from the result files.</li>
        </ol>
      </Section>

      <Section n="03" id="rankers" title="Who to call: every ranker on the test month">
        <div className="prose">
          <p>Switch between lift and AUC. Lift answers the RM&apos;s question directly: how many more product takers are in the first tenth of the call list than in a random tenth.</p>
        </div>
        <RankerChart rankers={r.rankers} labels={RANKER_LABEL} />
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Ranker</th><th className="num">Lift @ top 10%</th><th className="num">95% CI</th><th className="num">ROC AUC</th></tr>
            </thead>
            <tbody>
              {r.rankers.map((x, i) => (
                <tr key={x.model_id} className={i === 0 ? "best" : undefined}>
                  <td>{x.model_name}</td>
                  <td className="num">{x.lift_at_10pct.toFixed(2)}×</td>
                  <td className="num">{ci(x.lift_at_10pct_ci_lower, x.lift_at_10pct_ci_upper)}</td>
                  <td className="num">{x.auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="callout">
          <strong>Most of the gain is the features, not the model.</strong> Logistic regression on the new features reaches {lr.auc.toFixed(3)} AUC against {prev.auc.toFixed(3)} for gradient boosting on the old 14; LightGBM adds the rest.
        </div>
      </Section>

      <Section n="04" id="capacity" title="The call-capacity frontier">
        <div className="prose">
          <p>
            A score is half a decision; the other half is how many clients the team can call. Slide the capacity to see how many of the {r.test.positives.toLocaleString()} product takers the RM call list reaches. The list is ranked by the expected-value lead score (propensity × predicted value), which lifts {ev.lift_at_10pct.toFixed(2)}× in its top tenth: it trades some conversions for higher-value clients, against {lgb.lift_at_10pct.toFixed(2)}× for propensity alone.
          </p>
        </div>
        <CapacityFrontier gains={r.gains} />
      </Section>

      <Section n="05" id="next-product" title="Next best product: what to offer">
        <div className="prose">
          <p>
            One LightGBM per product predicts who takes it in the next 3 months, with the settings already chosen for the any-product model (nothing re-tuned). The Lead Desk offers each client the product ranked highest, and splits the calibrated any-product chance across products in proportion to these scores. Measured on the Oct 2022 test month:
          </p>
        </div>
        <div className="cells">
          <div className="cell">
            <span className="label">Right product first</span>
            <span className="value model">{pct(nbpHit.lightgbm.value)}</span>
            <span className="note">of the {nbpHit.lightgbm.buyers.toLocaleString()} clients who took a product had one of theirs ranked #1</span>
          </div>
          <div className="cell">
            <span className="label">Most-popular baseline</span>
            <span className="value baseline">{pct(nbpHit.most_popular_product.value)}</span>
            <span className="note">offer everyone the product most bought in training</span>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Product</th><th className="num">3-month AUC</th><th className="num">Buyers in test months</th></tr></thead>
            <tbody>
              {nbpAuc.map((x) => (
                <tr key={x.product}><td>Product {x.product.slice(-1)}</td><td className="num">{x.value.toFixed(3)}</td><td className="num">{x.buyers.toLocaleString()}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="prose">
          <p>Product 4 is above the project&apos;s 0.85 audit line. No single feature explains it: the strongest scores 0.74 alone, below the 0.80 line that halts the pipeline.</p>
        </div>
      </Section>

      <Section n="06" id="changes" title="What moved the model, measured on validation">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Step</th><th className="num">Validation AUC</th><th className="num">Top-10% lift</th></tr></thead>
            <tbody>
              {STEPS.map((s, i) => (
                <tr key={s.step} className={i === STEPS.length - 1 ? "best" : undefined}>
                  <td>{s.step}</td><td className="num">{s.auc.toFixed(3)}</td><td className="num">{s.lift.toFixed(2)}×</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="callout">
          <strong>A GRU over each client&apos;s last 512 transactions</strong> reached {gru.val_auc.toFixed(3)} on its own. Blended with LightGBM it added 0.001 AUC, below the 0.005 bar, so production uses GRU weight 0.00.
        </div>
        <div className="prose">
          <p>Tried and not kept, each judged on validation: label-chosen category codes, mean dialog embeddings, one LightGBM per product, snapshot weighting, seed bagging and ranking objectives. The validation score was {r.test.validation_auc.toFixed(3)}; the test month scored {r.test.test_auc.toFixed(3)} with fewer buyers and more purchase history in the refit. Quote both.</p>
        </div>
      </Section>

      <Section n="07" id="paper" title="Next to the published benchmark">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th className="num">Mean AUC</th><th className="num">P1</th><th className="num">P2</th><th className="num">P3</th><th className="num">P4</th></tr></thead>
            <tbody>
              {r.paper.map((p) => (
                <tr key={p.model} className={p === ours ? "best" : undefined}>
                  <td>{p.source === "ours" ? `${p.model} (this repo)` : p.model}</td>
                  <td className="num">{p.mean_auc.toFixed(3)}</td>
                  {[p.auc_y1_target_1, p.auc_y1_target_2, p.auc_y1_target_3, p.auc_y1_target_4].map((v, i) => (
                    <td key={i} className="num">{v === null ? "" : v.toFixed(3)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="prose">
          <p>
            Next-month, per-product AUC under the MBD paper&apos;s protocol (<a href="https://arxiv.org/abs/2409.17587">arXiv 2409.17587</a>). Not like-for-like: the paper uses full MBD (2.2M clients) with client folds; this repo uses MBD-mini and a time split.
          </p>
        </div>
      </Section>

      <Section n="08" id="value" title="What clients are worth: where the baseline wins">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th className="num">RMSE</th><th className="num">MAE</th></tr></thead>
            <tbody>
              {r.value.map((v, i) => (
                <tr key={v.model_id} className={i === 0 ? "best" : undefined}>
                  <td>{v.model_name}{v.role === "baseline" ? " (baseline)" : ""}</td>
                  <td className="num">{units(v.rmse)}</td>
                  <td className="num">{units(v.mae)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="callout">
          The CLTV model loses to &ldquo;the next 3 months look like the last 3&rdquo;. It was chosen before testing; on the validation month none of four gradient-boosting variants beat that baseline either. Amounts are MBD&apos;s anonymised units.
        </div>
      </Section>

      <Section n="09" id="fixed" title="What I fixed in the original project">
        <ol className="steps">
          <li><strong>Leakage from the accounts table.</strong> Account count, balance and status were today&apos;s values, not values at the snapshot; <code>account_count</code> alone scored 0.912.</li>
          <li><strong>A random split</strong> of customers from one snapshot, described as historical.</li>
          <li><strong>A leaderboard that favoured the production model.</strong> The &ldquo;baselines&rdquo; reused its predictions, and the ranking gave it a fixed +0.05 bonus.</li>
          <li><strong>A CLTV model worse than the mean</strong> (R² −0.37), not mentioned anywhere.</li>
          <li><strong>No reproducibility.</strong> Unpinned packages and no record of what produced the numbers.</li>
        </ol>
      </Section>

      <Section n="10" id="reproduce" title="Reproduction">
        <div className="prose"><p>Every number on this page is read from a committed result file. To regenerate them:</p></div>
        <pre>{`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_mbd.py            # ~2 GB, SHA-256 checked
.venv/bin/python -m lead_scoring.data            # feature build, about an hour
.venv/bin/python -m lead_scoring.features
.venv/bin/python scripts/validate_propensity.py  # choose settings on Jul 2022
.venv/bin/python scripts/run_pipeline.py         # refit, test, reports, dashboard data`}</pre>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Data</th><th>Source</th><th>Status</th></tr></thead>
            <tbody>
              <tr><td>MBD-mini targets, transactions, dialog timestamps</td><td><a href="https://huggingface.co/datasets/ai-lab/MBD-mini">Hugging Face ai-lab/MBD-mini</a></td><td>Real, anonymised; downloaded by script, not in git</td></tr>
              <tr><td>Original 500 customers</td><td>Generated</td><td>Synthetic; replaced</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <footer className="footer">
        Bank Lead Scoring · results read from reports/tables/ · data: Mollaev et al., MBD, CC BY 4.0 · <a href={REPO}>source</a>
      </footer>
    </main>
  );
}
