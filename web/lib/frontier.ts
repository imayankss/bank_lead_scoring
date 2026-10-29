export type GainsRow = {
  decile: number;     // 10 = top-scored tenth
  n: number;          // clients in the decile
  converters: number; // clients who took a product in the next 3 months
  conversion_rate: number;
  lift: number;
  cumulative_share_of_converters: number;
};

export type Readout = {
  clients: number;    // clients called
  buyers: number;     // product takers among them
  share: number;      // share of all product takers reached, 0-1
  hitRate: number;    // buyers / called, 0-1
  lift: number;       // hit rate vs calling clients at random
};

/** What an RM team that calls the top `k` deciles (1-10), rows ordered best-scored first, would get. */
export function readout(gains: GainsRow[], k: number): Readout | null {
  if (gains.length === 0) return null;
  // Cumulative over the top k deciles: a team calls the whole list, not just the k-th slice.
  const top = gains.slice(0, k);
  const clients = top.reduce((s, g) => s + g.n, 0);
  const buyers = top.reduce((s, g) => s + g.converters, 0);
  const allClients = gains.reduce((s, g) => s + g.n, 0);
  const allBuyers = gains.reduce((s, g) => s + g.converters, 0);
  const hitRate = buyers / clients;
  return { clients, buyers, share: buyers / allBuyers, hitRate, lift: hitRate / (allBuyers / allClients) };
}
