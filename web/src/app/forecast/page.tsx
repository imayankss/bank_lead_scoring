import { Activity, IndianRupee, LineChart, Percent } from "lucide-react";
import { ActualVsPredictedChart, MultiScenarioChart } from "@/components/charts/ForecastChart";
import { AnimatedPage } from "@/components/dashboard/AnimatedPage";
import { ChartCard } from "@/components/dashboard/ChartCard";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { SectionHeader } from "@/components/dashboard/SectionHeader";
import { currency, percent } from "@/lib/formatters";
import { getForecast } from "@/lib/data";

export default async function ForecastPage() {
  const forecast = await getForecast();
  return (
    <AnimatedPage>
      <SectionHeader eyebrow="Forecast Explorer" title="Actuals, Predictions, and Portfolio Horizon">
        Forecast visuals are exported from backend artifacts and scenario analysis.
      </SectionHeader>
      <div className="mb-6 grid gap-4 md:grid-cols-3">
        <MetricCard label="Expected value" value={currency(forecast.latest.total_expected_value)} icon={IndianRupee} tone="mint" />
        <MetricCard label="Average CLTV" value={currency(forecast.latest.average_cltv)} icon={LineChart} tone="cyan" />
        <MetricCard label="Average propensity" value={percent((forecast.latest.average_propensity ?? 0) * 100)} icon={Percent} tone="amber" />
      </div>
      <div className="grid gap-5 xl:grid-cols-2">
        <ChartCard title="Actual vs Predicted CLTV">
          <ActualVsPredictedChart data={forecast.actual_vs_predicted} />
        </ChartCard>
        <ChartCard title="Scenario Confidence Band">
          <MultiScenarioChart data={forecast.forecast} />
        </ChartCard>
      </div>
      <div className="mt-5 glass rounded-lg p-5">
        <div className="mb-4 flex items-center gap-2 text-white">
          <Activity className="h-5 w-5 text-cyan" />
          Latest Forecast Records
        </div>
        <div className="max-h-96 overflow-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-[0.16em] text-slate-500">
              <tr>
                <th className="py-3">Customer</th>
                <th>Actual</th>
                <th>Predicted CLTV</th>
                <th>Expected Value</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {forecast.actual_vs_predicted.slice(0, 20).map((row, index) => (
                <tr key={`${row.customer_id}-${index}`} className="text-slate-300">
                  <td className="py-3 text-white">{row.customer_id}</td>
                  <td>{currency(row.actual_future_revenue_12m)}</td>
                  <td>{currency(row.predicted_cltv)}</td>
                  <td>{currency(row.expected_value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </AnimatedPage>
  );
}
