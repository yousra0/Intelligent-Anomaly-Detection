"use client";

import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Building2, Calendar, ArrowRight, User, Trash2 } from "lucide-react";
import {
  formatDate,
  STATUS_LABELS_FR,
  MISSION_TYPE_LABELS_FR,
} from "@/lib/utils";
import { usePermissions } from "@/lib/hooks/usePermissions";
import { missionService } from "@/lib/api/missionService";
import { userService } from "@/lib/api/userService";
import type { Mission } from "@/types";

interface MissionCardProps {
  mission: Mission;
}

export function MissionCard({ mission }: MissionCardProps) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { can } = usePermissions();

  const { data: auditors = [] } = useQuery({
    queryKey: ["users", "auditors"],
    queryFn: userService.getAuditors,
  });

  const assignedAuditor = mission.assigned_to
    ? auditors.find((a) => a.id === mission.assigned_to)
    : null;

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

        {/* Assigned auditor */}
        {assignedAuditor && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <User className="h-3.5 w-3.5 shrink-0 text-pwc-orange" />
            <span className="text-xs">
              <span className="text-muted-foreground">Assigné à : </span>
              <span className="font-medium text-foreground">{assignedAuditor.name}</span>
            </span>
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
            className="shrink-0 text-red-500 hover:border-red-300 hover:bg-red-50"
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
