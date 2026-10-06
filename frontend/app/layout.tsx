import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Supply Chain Intelligence Platform",
  description: "Demand forecasting, store clustering, and purchase order recommendations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, padding: 0 }}>
        <nav style={{ padding: "1rem 2rem", borderBottom: "1px solid #e5e5e5" }}>
          <strong>Supply Chain Platform</strong>
          <span style={{ marginLeft: "2rem" }}>
            <a href="/dashboard" style={{ marginRight: "1rem" }}>Dashboard</a>
            <a href="/forecasts" style={{ marginRight: "1rem" }}>Forecasts</a>
            <a href="/clusters" style={{ marginRight: "1rem" }}>Clusters</a>
            <a href="/purchase-orders" style={{ marginRight: "1rem" }}>Purchase Orders</a>
            <a href="/reports">Reports</a>
          </span>
        </nav>
        <main style={{ padding: "2rem" }}>{children}</main>
      </body>
    </html>
  );
}
