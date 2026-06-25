import { CalendarDays, TrendingUp } from "lucide-react";
import { MonthPatternChart, TrendChart } from "@/components/charts/SeasonalityCharts";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { getSeasonality } from "@/lib/data";

export default async function SeasonalityPage() {
  const seasonality = await getSeasonality();
  const latest = seasonality.monthly[seasonality.monthly.length - 1] ?? {};
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Seasonality" title="Transaction Trend and Recurring Patterns">
        Monthly transaction behavior gives context to current customer activity and expected value.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-2">
        <MetricCard label="Monthly periods" value={seasonality.monthly.length.toString()} icon={CalendarDays} tone="cyan" />
        <MetricCard label="Latest active customers" value={String(latest.active_customers ?? 0)} icon={TrendingUp} tone="mint" />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <ChartCard title="Transaction Amount Trend">
          <TrendChart data={seasonality.monthly} />
        </ChartCard>
        <ChartCard title="Average Monthly Seasonality">
          <MonthPatternChart data={seasonality.pattern} />
        </ChartCard>
      </div>
    </AnimatedPage>
  );
}
