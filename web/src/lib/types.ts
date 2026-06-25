export type LeadCategory = "Hot" | "Medium" | "Cold" | string;

export interface Kpis {
  customers: number;
  features: number;
  models_trained: number;
  best_model: string;
  best_rmse: number;
  forecast_confidence: number;
  total_expected_value: number;
}

export interface LeadScore {
  customer_id: string;
  predicted_cltv: number;
  predicted_propensity: number;
  expected_value: number;
  lead_score: number;
  lead_category: LeadCategory;
}

export interface OverviewData {
  project: { name: string; description: string; domain: string };
  kpis: Kpis;
  lead_categories: Record<string, number>;
  top_leads: LeadScore[];
  decile_lift: Array<Record<string, number>>;
}

export interface ForecastPoint {
  customer_id?: string;
  actual_future_revenue_12m?: number;
  actual_converted_12m?: number;
  predicted_cltv?: number;
  predicted_propensity?: number;
  expected_value?: number;
  month?: number;
  scenario?: string;
  monthly_expected_value?: number;
  cumulative_expected_value?: number;
}

export interface ForecastData {
  actual_vs_predicted: ForecastPoint[];
  forecast: ForecastPoint[];
  latest: Record<string, number>;
}

export interface ModelRecord {
  model_id: string;
  model_name?: string;
  display_name?: string;
  role?: string;
  description?: string;
  mae?: number;
  rmse?: number;
  mape?: number;
  smape?: number;
  r2?: number;
  auc?: number;
  business_score?: number;
  rank?: number;
  is_best?: boolean;
}

export interface ModelsData {
  registry: ModelRecord[];
  leaderboard: ModelRecord[];
  metrics: Record<string, number>;
  model_metadata: Record<string, unknown>;
  feature_importance: Array<Record<string, number | string>>;
}

export interface DiagnosticsData {
  summary: Record<string, number>;
  residuals: Array<Record<string, number | string>>;
  feature_importance: Array<Record<string, number | string>>;
}

export interface RisksData {
  summary: Record<string, number | string | Array<Record<string, unknown>>>;
  signals: Array<Record<string, number | string>>;
}

export interface SeasonalityData {
  monthly: Array<Record<string, number | string>>;
  pattern: Array<Record<string, number>>;
}

export interface ScenarioData {
  base_expected_value: number;
  scenarios: ForecastPoint[];
  sensitivity: Array<Record<string, number>>;
}

export interface MethodologyData {
  pipeline: string[];
  leakage_prevention: string[];
  limitations: string[];
  dataset: {
    training_rows: number;
    positive_customers: number;
    feature_columns: string[];
  };
}
