import { promises as fs } from "fs";
import path from "path";
import type {
  DiagnosticsData,
  ForecastData,
  MethodologyData,
  ModelsData,
  OverviewData,
  RisksData,
  ScenarioData,
  SeasonalityData
} from "./types";

const dataDir = path.join(process.cwd(), "public", "data");

async function readJson<T>(file: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(path.join(dataDir, file), "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export const getOverview = () =>
  readJson<OverviewData>("overview.json", {
    project: { name: "Lead Scoring Intelligence", description: "", domain: "" },
    kpis: {
      customers: 0,
      features: 0,
      models_trained: 0,
      best_model: "Unavailable",
      best_rmse: 0,
      forecast_confidence: 0,
      total_expected_value: 0
    },
    lead_categories: {},
    top_leads: [],
    decile_lift: []
  });

export const getForecast = () => readJson<ForecastData>("forecast.json", { actual_vs_predicted: [], forecast: [], latest: {} });
export const getModels = () => readJson<ModelsData>("models.json", { registry: [], leaderboard: [], metrics: {}, model_metadata: {}, feature_importance: [] });
export const getDiagnostics = () => readJson<DiagnosticsData>("diagnostics.json", { summary: {}, residuals: [], feature_importance: [] });
export const getRisks = () => readJson<RisksData>("risks.json", { summary: {}, signals: [] });
export const getSeasonality = () => readJson<SeasonalityData>("seasonality.json", { monthly: [], pattern: [] });
export const getScenarios = () => readJson<ScenarioData>("scenarios.json", { base_expected_value: 0, scenarios: [], sensitivity: [] });
export const getMethodology = () =>
  readJson<MethodologyData>("methodology.json", {
    pipeline: [],
    leakage_prevention: [],
    limitations: [],
    dataset: { training_rows: 0, positive_customers: 0, feature_columns: [] }
  });
