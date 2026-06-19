"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { missionService } from "@/lib/api/missionService";
import { userService } from "@/lib/api/userService";
import { MissionCard } from "@/components/missions/MissionCard";
import { CreateMissionModal } from "@/components/missions/CreateMissionModal";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, Search, FolderOpen } from "lucide-react";
import { usePermissions } from "@/lib/hooks/usePermissions";
import type { MissionStatus } from "@/types";

export default function MissionsPage() {
  const [createOpen, setCreateOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<MissionStatus | "all">("all");

  const { can, user } = usePermissions();
  const { t } = useLanguage();
  const canCreate = can("mission.create");
  const canViewAll = can("mission.view_all");

  const { data: missions = [], isLoading, error } = useQuery({
    queryKey: ["missions"],
    queryFn: missionService.getAll,
    staleTime: 2 * 60 * 1000,
  });

  const { data: auditors = [] } = useQuery({
    queryKey: ["users", "auditors"],
    queryFn: userService.getAuditors,
    staleTime: 5 * 60 * 1000,
  });

  const visibleMissions = canViewAll
    ? missions
    : missions.filter((m) => m.assigned_to === user?.id);

  const filtered = visibleMissions.filter((m) => {
    const matchSearch =
      !search ||
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.company_name.toLowerCase().includes(search.toLowerCase());
    const matchStatus = statusFilter === "all" || m.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const STATUSES: (MissionStatus | "all")[] = ["all", "active", "in_progress", "completed", "archived"];

  const countLabel = visibleMissions.length === 1
    ? t("missions.countSingular", { count: visibleMissions.length })
    : t("missions.countPlural", { count: visibleMissions.length });

  return (
    <>
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">{t("missions.title")}</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">{countLabel}</p>
        </div>
        {canCreate && (
          <Button className="gap-2 self-start" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            {t("missions.create")}
          </Button>
        )}
      </div>

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={t("missions.searchPlaceholder")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={statusFilter} onValueChange={(v) => setStatusFilter(v as MissionStatus | "all")}>
          <SelectTrigger className="w-[180px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUSES.map((s) => (
              <SelectItem key={s} value={s}>
                {s === "all" ? t("status.all") : t(`status.${s}`)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-52 rounded-lg" />)}
        </div>
      ) : error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-center text-red-700">
          <p className="font-medium">{t("missions.loadError")}</p>
          <p className="mt-1 text-sm">{t("missions.loadErrorDesc")}</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-card py-16 text-center">
          <FolderOpen className="h-12 w-12 text-muted-foreground/40" />
          <p className="mt-4 font-medium text-muted-foreground">
            {search || statusFilter !== "all"
              ? t("missions.noResults")
              : canCreate
              ? t("missions.noMissions")
              : t("missions.noAssigned")}
          </p>
          {!search && statusFilter === "all" && canCreate && (
            <Button className="mt-4 gap-2" onClick={() => setCreateOpen(true)}>
              <Plus className="h-4 w-4" />
              {t("missions.createFirst")}
            </Button>
          )}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 animate-fade-in">
          {filtered.map((mission) => (
            <MissionCard key={mission.id} mission={mission} auditors={auditors} />
          ))}
        </div>
      )}

      {canCreate && (
        <CreateMissionModal open={createOpen} onClose={() => setCreateOpen(false)} />
      )}
    </>
  );
}
