import { BrainCircuit, Medal, Target } from "lucide-react";
import { ModelMetricChart } from "@/components/charts/ModelCharts";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ModelLeaderboard } from "@/components/dashboard/ModelLeaderboard";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { currency, decimal } from "@/lib/formatters";
import { getModels } from "@/lib/data";

export default async function ModelsPage() {
  const models = await getModels();
  const best = models.leaderboard.find((model) => model.is_best) ?? models.leaderboard[0];
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Model Leaderboard" title="Model Battle Arena">
        Models and baselines compete on business-oriented lead prioritization metrics.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <MetricCard label="Best model" value={best?.model_name ?? "Unavailable"} icon={Medal} tone="mint" />
        <MetricCard label="Business score" value={decimal(best?.business_score, 3)} icon={Target} tone="cyan" />
        <MetricCard label="Best RMSE" value={currency(best?.rmse)} icon={BrainCircuit} tone="amber" />
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.05fr_0.95fr]">
        <ChartCard title="Business Score Comparison">
          <ModelMetricChart data={models.leaderboard} />
        </ChartCard>
        <div className="glass rounded-lg p-5">
          <h3 className="mb-4 text-base font-semibold text-white">Leaderboard</h3>
          <ModelLeaderboard models={models.leaderboard} />
        </div>
      </div>
      <div className="mt-5 grid gap-4 md:grid-cols-3">
        {models.registry.map((entry) => (
          <div key={entry.model_id} className="glass rounded-lg p-5">
            <p className="text-sm text-cyan">{entry.role}</p>
            <h3 className="mt-2 text-lg font-semibold text-white">{entry.display_name}</h3>
            <p className="mt-3 text-sm leading-6 text-slate-400">{entry.description}</p>
          </div>
        ))}
      </div>
    </AnimatedPage>
  );
}
