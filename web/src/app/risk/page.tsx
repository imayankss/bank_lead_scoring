import { AlertTriangle, ShieldAlert, ShieldCheck } from "lucide-react";
import { RiskDistributionChart } from "@/components/charts/RiskCharts";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { RiskBadge } from "@/components/dashboard/RiskBadge";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { decimal } from "@/lib/formatters";
import { getRisks } from "@/lib/data";

export default async function RiskPage() {
  const risks = await getRisks();
  const summary = risks.summary;
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Risk Intelligence" title="Outreach Risk and Confidence">
        Risk combines stale activity, account status, uncertainty, credit score signals, and fallback rules.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-4">
        <MetricCard label="Average confidence" value={decimal(Number(summary.average_confidence), 1)} icon={ShieldCheck} tone="mint" />
        <MetricCard label="Composite risk" value={decimal(Number(summary.overall_risk_score), 1)} icon={ShieldAlert} tone="amber" />
        <MetricCard label="High risk" value={String(summary.high_risk_customers ?? 0)} icon={AlertTriangle} tone="rose" />
        <MetricCard label="Low risk" value={String(summary.low_risk_customers ?? 0)} icon={ShieldCheck} tone="cyan" />
      </div>
      <div className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
        <ChartCard title="Risk Distribution">
          <RiskDistributionChart data={risks.signals} />
        </ChartCard>
        <div className="glass rounded-lg p-5">
          <h3 className="mb-4 text-base font-semibold text-white">Priority Alerts</h3>
          <div className="space-y-3">
            {risks.signals.slice(0, 8).map((row) => (
              <div key={String(row.customer_id)} className="flex items-center justify-between gap-4 rounded-lg border border-line bg-white/[0.03] p-4">
                <div>
                  <div className="font-medium text-white">{row.customer_id}</div>
                  <div className="mt-1 text-sm text-slate-500">{row.alert}</div>
                </div>
                <RiskBadge category={String(row.risk_category)} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </AnimatedPage>
  );
}
