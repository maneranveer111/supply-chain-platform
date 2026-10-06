"use client";

import { useEffect, useState } from "react";
import { api, ForecastResponse } from "@/lib/api-client";

export default function DashboardPage() {
  const [storeId, setStoreId] = useState(1);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getForecast(storeId)
      .then((data) => {
        if (!cancelled) setForecast(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [storeId]);

  return (
    <div>
      <h1>Dashboard</h1>

      <label>
        Store ID:{" "}
        <input
          type="number"
          value={storeId}
          onChange={(e) => setStoreId(Number(e.target.value))}
          style={{ width: "80px" }}
        />
      </label>

      {loading && <p>Loading forecast...</p>}
      {error && <p style={{ color: "red" }}>Error: {error}</p>}

      {forecast && (
        <table style={{ marginTop: "1rem", borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr>
              <th style={cellStyle}>Day</th>
              <th style={cellStyle}>Date</th>
              <th style={cellStyle}>Predicted Sales</th>
            </tr>
          </thead>
          <tbody>
            {forecast.forecast.map((day) => (
              <tr key={day.day}>
                <td style={cellStyle}>{day.day}</td>
                <td style={cellStyle}>{day.date}</td>
                <td style={cellStyle}>{day.predicted_sales.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const cellStyle: React.CSSProperties = {
  border: "1px solid #e5e5e5",
  padding: "0.5rem 1rem",
  textAlign: "left",
};
