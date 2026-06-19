"use client";

import { useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Building2, Calendar, ArrowRight, Trash2 } from "lucide-react";
import {
  formatDate,
  STATUS_LABELS_FR,
  MISSION_TYPE_LABELS_FR,
} from "@/lib/utils";
import { usePermissions } from "@/lib/hooks/usePermissions";
import { missionService } from "@/lib/api/missionService";
import type { Mission, User as UserType } from "@/types";

interface MissionCardProps {
  mission: Mission;
  auditors: UserType[];
}

export function MissionCard({ mission, auditors }: MissionCardProps) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { can } = usePermissions();

  // Resolve all assigned auditors (multi-auditor support)
  const assignedAuditors: UserType[] = (() => {
    const ids = mission.assigned_auditors?.length
      ? mission.assigned_auditors
      : mission.assigned_to
      ? [mission.assigned_to]
      : [];
    return ids
      .map((id) => auditors.find((a) => a.id === id))
      .filter(Boolean) as UserType[];
  })();

  const deleteMutation = useMutation({
    mutationFn: () => missionService.delete(mission.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["missions"] });
      toast.success("Mission supprimée.");
    },
    onError: () => {
      toast.error("Erreur lors de la suppression.");
    },
  });

  return (
    <Card className="group flex flex-col transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <h3 className="font-semibold text-base leading-tight line-clamp-2">
            {mission.name}
          </h3>
          <Badge
            variant={mission.status as "active" | "in_progress" | "completed" | "archived"}
            className="shrink-0"
          >
            {STATUS_LABELS_FR[mission.status]}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="flex-1 space-y-3 text-sm">
        {/* Company */}
        <div className="flex items-center gap-2 text-muted-foreground">
          <Building2 className="h-3.5 w-3.5 shrink-0 text-pwc-orange" />
          <span className="font-medium text-foreground">{mission.company_name}</span>
        </div>

        {/* Type badge */}
        <div className="flex items-center gap-2">
          <span className="rounded bg-accent px-2 py-0.5 text-xs font-medium text-accent-foreground">
            {MISSION_TYPE_LABELS_FR[mission.mission_type]}
          </span>
        </div>

        {/* Dates */}
        <div className="flex items-center gap-2 text-muted-foreground">
          <Calendar className="h-3.5 w-3.5 shrink-0" />
          <span>
            {formatDate(mission.start_date)} → {formatDate(mission.end_date)}
          </span>
        </div>

        {/* Assigned auditors */}
        {assignedAuditors.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">
              Auditeur{assignedAuditors.length > 1 ? "s" : ""} assigné{assignedAuditors.length > 1 ? "s" : ""} :
            </span>
            <div className="flex flex-wrap gap-1">
              {assignedAuditors.map((a) => (
                <span
                  key={a.id}
                  className="rounded-full bg-pwc-orange/10 px-2 py-0.5 text-xs font-medium text-pwc-orange"
                >
                  {a.name}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Description */}
        {mission.description && (
          <p className="text-muted-foreground text-xs line-clamp-2 leading-relaxed">
            {mission.description}
          </p>
        )}
      </CardContent>

      <CardFooter className="gap-2 pt-3">
        <Button
          className="flex-1 gap-2 group-hover:bg-pwc-orange-dark"
          onClick={() => router.push(`/missions/${mission.id}`)}
        >
          Ouvrir
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </Button>
        {can("mission.delete") && (
          <Button
            variant="outline"
            size="icon"
            className="shrink-0 text-red-500 hover:border-red-300 hover:bg-red-50 dark:hover:bg-red-900/20"
            disabled={deleteMutation.isPending}
            onClick={() => deleteMutation.mutate()}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}
