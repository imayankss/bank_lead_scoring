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
  const rankers = models.leaderboard.filter((model) => model.task === "propensity");
  const best = rankers.find((model) => model.is_best) ?? rankers[0];
  const bestValue = models.leaderboard.find((model) => model.task === "value" && model.is_best);
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Model Leaderboard" title="Model Battle Arena">
        Every model and baseline is scored on the same later test months (Nov 2022 to Jan 2023), ranked by
        conversion lift in the top 10% of clients, with 95% bootstrap intervals. No candidate gets a bonus.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <MetricCard label="Best model" value={best?.model_name ?? "Unavailable"} icon={Medal} tone="mint" />
        <MetricCard
          label="Top-10% lift"
          value={`${decimal(best?.lift_at_10pct, 2)}x`}
          detail={`95% CI ${decimal(best?.lift_at_10pct_ci_lower, 2)}–${decimal(best?.lift_at_10pct_ci_upper, 2)}`}
          icon={Target}
          tone="cyan"
        />
        <MetricCard label="Best value model (RMSE)" value={bestValue?.model_name ?? "Unavailable"} detail={currency(bestValue?.rmse)} icon={BrainCircuit} tone="amber" />
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.05fr_0.95fr]">
        <ChartCard title="Top-10% Lift by Model (1.0 = random)">
          <ModelMetricChart data={rankers} />
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
