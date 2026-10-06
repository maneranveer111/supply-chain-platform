"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api-client";

export default function ClustersPage() {
  const [clusters, setClusters] = useState<unknown[]>([]);

  useEffect(() => {
    api.getClusters().then(setClusters).catch(console.error);
  }, []);

  return (
    <div>
      <h1>Store Clusters</h1>
      <pre>{JSON.stringify(clusters, null, 2)}</pre>
    </div>
  );
}
