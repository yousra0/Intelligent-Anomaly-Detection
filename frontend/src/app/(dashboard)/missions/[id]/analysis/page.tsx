"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { missionService } from "@/lib/api/missionService";
import { AnalysisWizard } from "@/components/analysis/AnalysisWizard";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowLeft } from "lucide-react";

export default function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const { data: mission, isLoading } = useQuery({
    queryKey: ["mission", id],
    queryFn: () => missionService.getById(id),
  });

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-56" />
        <Skeleton className="h-96 rounded-lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <button
          className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          onClick={() => router.push(`/missions/${id}`)}
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          {mission?.name ?? "Mission"}
        </button>
        <h1 className="mt-1 text-xl font-bold">Analyse de détection d'anomalies</h1>
        {mission && (
          <p className="text-sm text-muted-foreground">
            {mission.company_name} · {mission.name}
          </p>
        )}
      </div>

      {/* Analysis wizard */}
      <AnalysisWizard
        missionId={id}
        missionName={mission?.name}
        companyName={mission?.company_name}
      />
    </div>
  );
}
