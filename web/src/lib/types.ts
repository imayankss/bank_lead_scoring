export type LeadCategory = "Hot" | "Medium" | "Cold" | string;

export interface Kpis {
  customers: number;
  features: number;
  models_trained: number;
  best_model: string;
  best_rmse: number;
  best_value_model: string;
  best_lift_at_10pct: number;
  best_lift_ci: [number, number];
  test_auc: number;
  test_auc_ci: [number, number];
  test_positive_rate: number;
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
  actual_future_value?: number;
  actual_converted?: number;
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
  task?: "propensity" | "value";
  display_name?: string;
  role?: string;
  description?: string;
  mae?: number;
  rmse?: number;
  mape?: number;
  smape?: number;
  r2?: number;
  auc?: number;
  auc_ci_lower?: number;
  auc_ci_upper?: number;
  pr_auc?: number;
  lift_at_10pct?: number;
  lift_at_10pct_ci_lower?: number;
  lift_at_10pct_ci_upper?: number;
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
    test_rows: number;
    positive_customers: number;
    feature_columns: string[];
  };
}
