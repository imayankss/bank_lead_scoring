import { GitBranch, IndianRupee, SlidersHorizontal } from "lucide-react";
import { MultiScenarioChart } from "@/components/charts/ForecastChart";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { ScenarioLab } from "@/components/dashboard/ScenarioLab";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { currency } from "@/lib/formatters";
import { getScenarios } from "@/lib/data";

export default async function ScenarioPage() {
  const scenarios = await getScenarios();
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Scenario Lab" title="Portfolio Value What-If Analysis">
        Base, optimistic, and pessimistic cases are exported from Python; the slider simulates frontend sensitivity.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <MetricCard label="Base expected value" value={currency(scenarios.base_expected_value)} icon={IndianRupee} tone="mint" />
        <MetricCard label="Scenario paths" value="3" icon={GitBranch} tone="cyan" />
        <MetricCard label="Sensitivity controls" value="Active" icon={SlidersHorizontal} tone="amber" />
      </div>
      <div className="mb-5">
        <ChartCard title="Base / Optimistic / Pessimistic">
          <MultiScenarioChart data={scenarios.scenarios} />
        </ChartCard>
      </div>
      <ScenarioLab data={scenarios} />
    </AnimatedPage>
  );
}
