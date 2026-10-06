"use client";

import { useEffect, useState } from "react";
import { api, PurchaseOrderRecommendation } from "@/lib/api-client";

export default function PurchaseOrdersPage() {
  const [recommendations, setRecommendations] = useState<PurchaseOrderRecommendation[]>([]);

  useEffect(() => {
    api.getPurchaseOrderRecommendations().then(setRecommendations).catch(console.error);
  }, []);

  return (
    <div>
      <h1>Purchase Order Recommendations</h1>
      <ul>
        {recommendations.map((rec) => (
          <li key={rec.po_id}>
            PO #{rec.po_id} — Store {rec.store_id}: recommend {rec.recommended_qty} units — {rec.reason}
          </li>
        ))}
      </ul>
    </div>
  );
}
