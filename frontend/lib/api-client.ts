const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem("access_token");
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export interface ForecastDay {
  day: number;
  date: string;
  predicted_sales: number;
}

export interface ForecastResponse {
  store_id: number;
  horizon: number;
  window_used: number;
  forecast: ForecastDay[];
  generated_at: string;
}

export interface PurchaseOrderRecommendation {
  po_id: number;
  store_id: number;
  forecasted_demand: number;
  current_inventory: number;
  safety_stock: number;
  recommended_qty: number;
  reason: string;
}

export const api = {
  login: (email: string, password: string) =>
    request<{ access_token: string; token_type: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  getForecast: (storeId: number, horizon = 7) =>
    request<ForecastResponse>(`/stores/${storeId}/forecast?horizon=${horizon}`),

  getClusters: () => request<unknown[]>("/stores/clusters"),

  getStoreCluster: (storeId: number) => request<unknown>(`/stores/${storeId}/cluster`),

  getPurchaseOrderRecommendations: (storeId?: number) =>
    request<PurchaseOrderRecommendation[]>(
      `/purchase-orders/recommendations${storeId ? `?store_id=${storeId}` : ""}`
    ),

  getWeeklySummary: () => request<{ summary: string; stats: unknown }>("/reports/weekly-summary"),
};
