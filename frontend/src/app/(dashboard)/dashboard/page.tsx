"use client";

import { useQuery } from "@tanstack/react-query";
import { missionService } from "@/lib/api/missionService";
import { auditLogService } from "@/lib/api/auditLogService";
import { analysisRunService } from "@/lib/api/analysisRunService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { formatDateTime, ROLE_LABELS_FR, AUDIT_ACTION_LABELS_FR, STATUS_LABELS_FR } from "@/lib/utils";
import { usePermissions } from "@/lib/hooks/usePermissions";
import {
  FolderOpen,
  BarChart3,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Shield,
  TrendingUp,
  Activity,
} from "lucide-react";
import Link from "next/link";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import type { MissionStatus } from "@/types";

const STATUS_COLORS_MAP: Record<MissionStatus, string> = {
  active:      "#008246",
  in_progress: "#D04A02",
  completed:   "#293854",
  archived:    "#9CA3AF",
};

export default function DashboardPage() {
  const { user, role } = usePermissions();

  const { data: missions = [], isLoading: loadingMissions } = useQuery({
    queryKey: ["missions"],
    queryFn: missionService.getAll,
  });

  const { data: auditLogs = [], isLoading: loadingLogs } = useQuery({
    queryKey: ["audit-logs"],
    queryFn: () => auditLogService.getAll(50),
  });

  const { data: analysisRuns = [], isLoading: loadingRuns } = useQuery({
    queryKey: ["analysis-runs"],
    queryFn: analysisRunService.getAll,
  });

  const totalMissions = missions.length;
  const activeMissions = missions.filter((m) => m.status === "active" || m.status === "in_progress").length;
  const completedMissions = missions.filter((m) => m.status === "completed").length;
  const totalAnalyses = analysisRuns.length;
  const totalAnomalies = analysisRuns.reduce((acc, r) => acc + (r.result?.n_fraud ?? 0), 0);

  // Pie chart data for mission statuses
  const statusCounts = (["active", "in_progress", "completed", "archived"] as MissionStatus[])
    .map((s) => ({
      name: STATUS_LABELS_FR[s],
      value: missions.filter((m) => m.status === s).length,
      color: STATUS_COLORS_MAP[s],
    }))
    .filter((s) => s.value > 0);

  const isLoading = loadingMissions || loadingLogs || loadingRuns;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Tableau de bord</h1>
        <p className="mt-0.5 text-sm text-muted-foreground">
          Bienvenue, <span className="font-medium">{user?.name}</span> —{" "}
          {role ? ROLE_LABELS_FR[role] : ""}
        </p>
      </div>

      {/* KPI cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KPICard
          title="Missions totales"
          value={isLoading ? null : totalMissions}
          icon={FolderOpen}
          color="text-pwc-orange"
          bg="bg-orange-50"
          href="/missions"
        />
        <KPICard
          title="Missions actives"
          value={isLoading ? null : activeMissions}
          icon={Activity}
          color="text-blue-600"
          bg="bg-blue-50"
          href="/missions"
        />
        <KPICard
          title="Analyses lancées"
          value={isLoading ? null : totalAnalyses}
          icon={BarChart3}
          color="text-green-600"
          bg="bg-green-50"
          href="/history"
        />
        <KPICard
          title="Anomalies détectées"
          value={isLoading ? null : totalAnomalies}
          icon={AlertTriangle}
          color="text-red-600"
          bg="bg-red-50"
          href="/history"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Mission status pie chart */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-pwc-orange" />
              Répartition des missions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loadingMissions ? (
              <Skeleton className="h-48 w-full rounded" />
            ) : statusCounts.length === 0 ? (
              <p className="py-12 text-center text-sm text-muted-foreground">
                Aucune mission
              </p>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={statusCounts}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {statusCounts.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v, n) => [v, n]} />
                  <Legend iconType="circle" iconSize={8} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Quick stats */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <CheckCircle2 className="h-4 w-4 text-pwc-orange" />
              Synthèse des analyses
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {loadingRuns ? (
              Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full rounded" />
              ))
            ) : analysisRuns.length === 0 ? (
              <p className="py-8 text-center text-sm text-muted-foreground">
                Aucune analyse effectuée
              </p>
            ) : (
              analysisRuns.slice(0, 5).map((run) => (
                <div
                  key={run.id}
                  className="flex items-center justify-between rounded-lg border border-border bg-muted/20 px-3 py-2 text-sm"
                >
                  <div className="min-w-0">
                    <p className="truncate font-medium">{run.mission_name}</p>
                    <p className="text-xs text-muted-foreground">{run.dataset_name}</p>
                  </div>
                  <div className="ml-3 shrink-0 text-right">
                    <p className="font-semibold text-red-600">
                      {run.result?.n_fraud ?? 0} anomalie(s)
                    </p>
                    <p className="text-[10px] text-muted-foreground">
                      {formatDateTime(run.completed_at ?? run.started_at)}
                    </p>
                  </div>
                </div>
              ))
            )}
            {analysisRuns.length > 5 && (
              <Link href="/history" className="block text-center text-xs text-pwc-orange hover:underline">
                Voir tout l'historique →
              </Link>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent audit activity */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-base">
              <Shield className="h-4 w-4 text-pwc-orange" />
              Activité récente
            </CardTitle>
            <Link href="/audit-trail" className="text-xs text-pwc-orange hover:underline">
              Voir la piste d'audit complète →
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          {loadingLogs ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full rounded" />
              ))}
            </div>
          ) : auditLogs.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">
              Aucune activité enregistrée
            </p>
          ) : (
            <div className="space-y-1.5">
              {auditLogs.slice(0, 10).map((log) => (
                <div
                  key={log.id}
                  className="flex items-start gap-3 rounded-lg px-3 py-2 text-sm hover:bg-muted/30"
                >
                  <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium">{log.user_name}</span>
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        {ROLE_LABELS_FR[log.user_role]}
                      </Badge>
                      <span className="text-muted-foreground">
                        — {AUDIT_ACTION_LABELS_FR[log.action]}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground truncate">{log.details}</p>
                  </div>
                  <span className="shrink-0 text-[10px] text-muted-foreground whitespace-nowrap">
                    {formatDateTime(log.timestamp)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function KPICard({
  title,
  value,
  icon: Icon,
  color,
  bg,
  href,
}: {
  title: string;
  value: number | null;
  icon: typeof FolderOpen;
  color: string;
  bg: string;
  href: string;
}) {
  return (
    <Link href={href}>
      <Card className="transition-shadow hover:shadow-md">
        <CardContent className="pt-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-muted-foreground">{title}</p>
              {value === null ? (
                <Skeleton className="mt-1 h-8 w-16 rounded" />
              ) : (
                <p className="mt-1 text-3xl font-bold">{value}</p>
              )}
            </div>
            <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${bg}`}>
              <Icon className={`h-5 w-5 ${color}`} />
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
