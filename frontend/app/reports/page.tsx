"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";

export default function ReportsPage() {
  const [summary, setSummary] = useState<string>("");

  useEffect(() => {
    api.getWeeklySummary().then((data) => setSummary(data.summary)).catch(console.error);
  }, []);

  return (
    <div>
      <h1>Weekly Summary</h1>
      <p>{summary || "Loading..."}</p>
    </div>
  );
}
