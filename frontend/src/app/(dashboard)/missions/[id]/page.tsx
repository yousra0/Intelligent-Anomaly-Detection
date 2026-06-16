"use client";

import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { missionService } from "@/lib/api/missionService";
import { DatasetSection } from "@/components/datasets/DatasetSection";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ArrowLeft,
  Building2,
  Calendar,
  ClipboardList,
  BarChart3,
  Info,
} from "lucide-react";
import {
  formatDate,
  STATUS_LABELS_FR,
  MISSION_TYPE_LABELS_FR,
} from "@/lib/utils";
import type { Dataset } from "@/types";

export default function MissionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const { data: mission, isLoading, error } = useQuery({
    queryKey: ["mission", id],
    queryFn: () => missionService.getById(id),
  });

  const handleAnalyze = (dataset: Dataset) => {
    router.push(`/missions/${id}/analysis?datasetId=${dataset.id}&datasetName=${encodeURIComponent(dataset.name)}`);
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 rounded-lg" />
        <Skeleton className="h-64 rounded-lg" />
      </div>
    );
  }

  if (error || !mission) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center">
        <p className="font-medium text-red-700">Mission introuvable.</p>
        <Button variant="outline" className="mt-4" onClick={() => router.push("/missions")}>
          Retour aux missions
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Back + header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <button
            className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
            onClick={() => router.push("/missions")}
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Missions
          </button>
          <h1 className="text-xl font-bold">{mission.name}</h1>
        </div>
        <Button
          className="gap-2 shrink-0"
          onClick={() => router.push(`/missions/${id}/analysis`)}
        >
          <BarChart3 className="h-4 w-4" />
          Lancer une analyse
        </Button>
      </div>

      {/* Mission info card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between text-base">
            <span className="flex items-center gap-2">
              <Info className="h-4 w-4 text-pwc-orange" />
              Informations de la mission
            </span>
            <Badge variant={mission.status as "active" | "in_progress" | "completed" | "archived"}>
              {STATUS_LABELS_FR[mission.status]}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Société</p>
              <p className="flex items-center gap-1.5 font-medium text-sm">
                <Building2 className="h-3.5 w-3.5 text-pwc-orange" />
                {mission.company_name}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Type de mission</p>
              <p className="font-medium text-sm">
                {MISSION_TYPE_LABELS_FR[mission.mission_type]}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-xs text-muted-foreground">Période</p>
              <p className="flex items-center gap-1.5 font-medium text-sm">
                <Calendar className="h-3.5 w-3.5 text-muted-foreground" />
                {formatDate(mission.start_date)} → {formatDate(mission.end_date)}
              </p>
            </div>
          </div>

          {mission.description && (
            <>
              <Separator className="my-4" />
              <div>
                <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <ClipboardList className="h-3.5 w-3.5" />
                  Description
                </p>
                <p className="text-sm leading-relaxed text-foreground">{mission.description}</p>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Datasets section */}
      <DatasetSection missionId={id} onAnalyze={handleAnalyze} />
    </div>
  );
}
