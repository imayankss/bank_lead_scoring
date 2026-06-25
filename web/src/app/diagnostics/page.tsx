import { Activity, AlertCircle, Gauge } from "lucide-react";
import { ErrorBucketChart, ResidualChart } from "@/components/charts/DiagnosticsCharts";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { currency, decimal } from "@/lib/formatters";
import { getDiagnostics } from "@/lib/data";

export default async function DiagnosticsPage() {
  const diagnostics = await getDiagnostics();
  const summary = diagnostics.summary;
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Diagnostics" title="Residual Pattern and Error Stability">
        Diagnostic panels reveal whether the value model is consistently over or under forecasting.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <MetricCard label="Mean absolute error" value={currency(summary.mean_absolute_error)} icon={Gauge} tone="amber" />
        <MetricCard label="Mean residual" value={currency(summary.mean_residual)} icon={Activity} tone="cyan" />
        <MetricCard label="Extreme errors" value={decimal(summary.extreme_error_count, 0)} icon={AlertCircle} tone="rose" />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <ChartCard title="Residual Trace">
          <ResidualChart data={diagnostics.residuals} />
        </ChartCard>
        <ChartCard title="Error Distribution">
          <ErrorBucketChart data={diagnostics.residuals} />
        </ChartCard>
      </div>
    </AnimatedPage>
  );
}
