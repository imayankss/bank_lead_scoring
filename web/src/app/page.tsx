import { BrainCircuit, Database, Gauge, Target, Users, Wallet } from "lucide-react";
import { LeadCategoryChart, ScenarioForecastChart } from "@/components/charts/ForecastChart";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { PipelineTimeline } from "@/components/dashboard/PipelineTimeline";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { currency, decimal, interval, percent } from "@/lib/formatters";
import { getForecast, getMethodology, getOverview } from "@/lib/data";

export default async function OverviewPage() {
  const [overview, forecast, methodology] = await Promise.all([getOverview(), getForecast(), getMethodology()]);
  const kpis = overview.kpis;

  return (
    <AnimatedPage>
      <section className="signal-field mb-6 overflow-hidden rounded-lg border border-line p-6 md:p-8">
        <div className="grid gap-6 xl:grid-cols-[1.25fr_0.75fr]">
          <div>
            <p className="mb-3 text-xs uppercase tracking-[0.24em] text-cyan">AI Analytics Command Center</p>
            <h1 className="max-w-4xl text-4xl font-semibold leading-tight text-white md:text-6xl">
              Lead Scoring Intelligence Platform
            </h1>
            <p className="mt-5 max-w-3xl text-base leading-7 text-slate-300">
              CLTV, propensity, risk signals, model diagnostics, and scenario intelligence for prioritizing banking customers.
            </p>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
              Built for a Bank of India-style relationship-manager workflow. Demonstrated on Sber AI Lab&apos;s public
              MBD-mini data (business clients, CC BY 4.0), not Bank of India customers. Every number is measured on
              later months than the model was trained on.
            </p>
          </div>
          <div className="rounded-lg border border-line bg-ink/55 p-5">
            <p className="text-sm text-slate-400">Best prioritization model (test months Nov 2022 to Jan 2023)</p>
            <div className="mt-3 text-2xl font-semibold text-white">{kpis.best_model}</div>
            <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-white/[0.05] p-3">
                <div className="text-slate-500">Lift in top 10%</div>
                <div className="mt-1 text-xl text-mint">{decimal(kpis.best_lift_at_10pct, 2)}x</div>
                <div className="mt-1 text-xs text-slate-500">{interval(kpis.best_lift_ci)}</div>
              </div>
              <div className="rounded-lg bg-white/[0.05] p-3">
                <div className="text-slate-500">AUC</div>
                <div className="mt-1 text-xl text-cyan">{decimal(kpis.test_auc, 3)}</div>
                <div className="mt-1 text-xs text-slate-500">{interval(kpis.test_auc_ci, 3)}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="mb-8 grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="Customers scored" value={kpis.customers.toLocaleString()} detail="Full scoring universe" icon={Users} tone="cyan" />
        <MetricCard label="Model features" value={kpis.features.toString()} detail="Leakage-safe inputs" icon={Database} tone="mint" />
        <MetricCard label="Models trained" value={kpis.models_trained.toString()} detail="CLTV and propensity" icon={BrainCircuit} tone="amber" />
        <MetricCard label="Expected value" value={currency(kpis.total_expected_value)} detail="Portfolio opportunity" icon={Wallet} tone="mint" />
        <MetricCard label="Base rate" value={percent((kpis.test_positive_rate ?? 0) * 100)} detail="Clients taking a product, test months" icon={Target} tone="rose" />
      </div>

      <div className="mb-8 grid gap-5 xl:grid-cols-2">
        <ChartCard title="Lead Category Distribution">
          <LeadCategoryChart data={overview.lead_categories} />
        </ChartCard>
        <ChartCard title="3 Month Portfolio Scenario">
          <ScenarioForecastChart data={forecast.forecast.filter((row) => row.scenario === "Base")} />
        </ChartCard>
      </div>

      <SectionHeader eyebrow="Pipeline" title="Verified ML Flow">
        The backend exports static JSON for the product dashboard, keeping model logic in Python.
      </SectionHeader>
      <PipelineTimeline steps={methodology.pipeline} />

      <div className="mt-8 grid gap-4 md:grid-cols-3">
        {overview.top_leads.slice(0, 3).map((lead) => (
          <div key={lead.customer_id} className="glass rounded-lg p-5">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm text-slate-400">{lead.customer_id}</span>
              <span className="rounded-full border border-mint/30 bg-mint/10 px-3 py-1 text-xs text-mint">{lead.lead_category}</span>
            </div>
            <div className="text-3xl font-semibold text-white">{lead.lead_score}/100</div>
            <div className="mt-3 text-sm text-slate-400">Expected value {currency(lead.expected_value)}</div>
            <div className="mt-1 text-sm text-slate-500">Propensity {decimal(lead.predicted_propensity * 100, 1)}%</div>
          </div>
        ))}
      </div>
    </AnimatedPage>
  );
}
