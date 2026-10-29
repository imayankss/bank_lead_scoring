export type Family = "product_history" | "conversations" | "transaction_mix" | "activity";

export type Client = {
  rank: number;
  id: string;
  score: number;
  tier: "Hot" | "Medium" | "Cold";
  p_any: number;
  p: number[];          // 3-month probability of each product, products 1-4
  next: number;         // next best product, 1-4
  value: number;        // predicted 3-month value (model estimate)
  ev: number;           // p_any × value: what the lead score ranks by
  vol90: number;        // actual transaction volume, last 90 days
  txn90: number;
  recency: number;      // days since the last transaction
  dialogs90: number;
  spend: number[];      // volume per 30-day window, oldest first
  months_since: (number | null)[];
  bought_before: number[];
  drivers: Record<Family, number>; // contribution to the propensity score, log-odds
  flags: string[];
};

export type Crm = {
  snapshot: string;
  summary: { clients: number; tiers: Record<string, number>; expected_takers: number; expected_value: number; shown: number };
  products: { product: number; next_for: number; expected_takers: number }[];
  test: { measure: string; product: string; model: string; value: number; buyers: number }[];
  clients: Client[];
};

// MBD does not name its products; a bank maps its own names here.
export const PRODUCT_LABEL: Record<number, string> = { 1: "Product 1", 2: "Product 2", 3: "Product 3", 4: "Product 4" };

export const FLAG_LABEL: Record<string, string> = {
  no_transactions_in_last_365d: "No transactions in a year",
  stale_transaction_activity: "No activity for 180+ days",
  activity_dropping: "Activity dropping"
};

export const compact = (v: number) => new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);
export const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

/** The three strongest reasons behind a client's propensity, in plain words, strongest first. */
export function reasons(c: Client): { text: string; up: boolean }[] {
  const latest = c.months_since.reduce<number | null>((best, m, i) => (m !== null && (best === null || m < c.months_since[best]!) ? i : best), null);
  const bought = c.bought_before.reduce((s, n) => s + n, 0);
  const recent = c.spend.at(-1)!;
  const usual = c.spend.reduce((s, v) => s + v, 0) / c.spend.length;

  const say: Record<Family, string> = {
    product_history:
      latest !== null
        ? `Bought ${PRODUCT_LABEL[latest + 1]} ${c.months_since[latest] === 0 ? "this month" : `${c.months_since[latest]} month${c.months_since[latest] === 1 ? "" : "s"} ago`}`
        : bought > 0 ? `Has taken ${bought} product${bought === 1 ? "" : "s"} before` : "No product purchases on record",
    conversations: c.dialogs90 > 0 ? `${c.dialogs90} conversation${c.dialogs90 === 1 ? "" : "s"} with the bank in 90 days` : "No conversations with the bank in 90 days",
    activity:
      usual > 0 && Math.abs(recent / usual - 1) >= 0.25
        ? `Last 30 days' volume ${recent > usual ? "up" : "down"} ${Math.abs(Math.round((recent / usual - 1) * 100))}% on the 6-month average`
        : `${c.txn90} transactions in 90 days, last one ${c.recency} day${c.recency === 1 ? "" : "s"} ago`,
    transaction_mix: "" // not a fact about the client, so its wording follows the direction below
  };
  return (Object.entries(c.drivers) as [Family, number][])
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([f, v]) => ({ text: f === "transaction_mix" ? `Mix of transaction types ${v > 0 ? "resembles" : "differs from"} past buyers` : say[f], up: v > 0 }));
}

/** A one-line suggestion for the RM; rules, not a model. */
export function nextStep(c: Client): string {
  const offer = PRODUCT_LABEL[c.next];
  if (c.flags.includes("no_transactions_in_last_365d") || c.flags.includes("stale_transaction_activity")) {
    return `Re-engage first (last transaction ${c.recency} days ago), then offer ${offer}.`;
  }
  if (c.months_since[c.next - 1] !== null && c.months_since[c.next - 1]! <= 2) {
    return `Offer ${offer} again: they bought it ${c.months_since[c.next - 1] === 0 ? "this month" : `${c.months_since[c.next - 1]} months ago`}.`;
  }
  return `Offer ${offer} (${pct(c.p[c.next - 1])} chance in the next 3 months).`;
}
