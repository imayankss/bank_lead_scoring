import { Database, FileWarning, ShieldCheck } from "lucide-react";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { PipelineTimeline } from "@/components/dashboard/PipelineTimeline";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { getMethodology } from "@/lib/data";

export default async function MethodologyPage() {
  const methodology = await getMethodology();
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Methodology" title="Data Flow, Leakage Prevention, and Limits">
        The dashboard presents generated artifacts; all model training and export logic stays in the Python backend.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <MetricCard label="Training rows" value={methodology.dataset.training_rows.toLocaleString()} icon={Database} tone="cyan" />
        <MetricCard label="Positive customers" value={methodology.dataset.positive_customers.toLocaleString()} icon={ShieldCheck} tone="mint" />
        <MetricCard label="Excluded fields" value={methodology.leakage_prevention.length.toString()} icon={FileWarning} tone="amber" />
      </div>
      <PipelineTimeline steps={methodology.pipeline} />
      <div className="mt-6 grid gap-5 xl:grid-cols-2">
        <section className="glass rounded-lg p-5">
          <h3 className="mb-4 text-base font-semibold text-white">Leakage Prevention</h3>
          <div className="flex flex-wrap gap-2">
            {methodology.leakage_prevention.map((field) => (
              <span key={field} className="rounded-full border border-cyan/20 bg-cyan/10 px-3 py-1 text-xs text-cyan">
                {field}
              </span>
            ))}
          </div>
        </section>
        <section className="glass rounded-lg p-5">
          <h3 className="mb-4 text-base font-semibold text-white">Limitations</h3>
          <ul className="space-y-3 text-sm leading-6 text-slate-400">
            {methodology.limitations.map((item) => (
              <li key={item} className="rounded-lg border border-line bg-white/[0.03] p-3">
                {item}
              </li>
            ))}
          </ul>
        </section>
      </div>
    </AnimatedPage>
  );
}
