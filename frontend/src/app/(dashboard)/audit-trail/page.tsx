"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { auditLogService } from "@/lib/api/auditLogService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  formatDate,
  formatDateTime,
  ROLE_LABELS_FR,
  AUDIT_ACTION_LABELS_FR,
} from "@/lib/utils";
import {
  Shield,
  Search,
  Download,
  Clock,
  User,
  List,
  GitBranch,
} from "lucide-react";
import type { AuditLog, AuditLogAction, UserRole } from "@/types";

const ACTION_ICON_COLOR: Record<string, string> = {
  login:                  "bg-green-100 text-green-700",
  logout:                 "bg-gray-100 text-gray-600",
  "mission.create":       "bg-blue-100 text-blue-700",
  "mission.update":       "bg-blue-50 text-blue-500",
  "mission.delete":       "bg-red-100 text-red-700",
  "mission.assign":       "bg-purple-100 text-purple-700",
  "dataset.upload":       "bg-orange-100 text-orange-700",
  "dataset.delete":       "bg-red-100 text-red-600",
  "dataset.replace":      "bg-yellow-100 text-yellow-700",
  "analysis.start":       "bg-cyan-100 text-cyan-700",
  "analysis.complete":    "bg-green-100 text-green-700",
  "report.generate":      "bg-indigo-100 text-indigo-700",
  "report.download":      "bg-indigo-50 text-indigo-500",
  "anomaly.comment":      "bg-gray-100 text-gray-600",
  "anomaly.status_change":"bg-yellow-100 text-yellow-700",
};

const ALL_ACTIONS = Object.keys(AUDIT_ACTION_LABELS_FR) as AuditLogAction[];

function exportCSV(logs: AuditLog[]) {
  const headers = ["Date", "Utilisateur", "Rôle", "Action", "Mission", "Détails"];
  const rows = logs.map((l) => [
    formatDateTime(l.timestamp),
    l.user_name,
    ROLE_LABELS_FR[l.user_role],
    AUDIT_ACTION_LABELS_FR[l.action],
    l.mission_name ?? "",
    l.details,
  ]);
  const csv = [headers, ...rows].map((r) => r.map((v) => `"${v}"`).join(";")).join("\n");
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `piste_audit_${formatDate(new Date().toISOString())}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function AuditTrailPage() {
  const [search, setSearch] = useState("");
  const [actionFilter, setActionFilter] = useState<AuditLogAction | "all">("all");
  const [roleFilter, setRoleFilter] = useState<UserRole | "all">("all");

  const { data: logs = [], isLoading } = useQuery({
    queryKey: ["audit-logs"],
    queryFn: () => auditLogService.getAll(500),
  });

  const filtered = logs.filter((l) => {
    const matchSearch =
      !search ||
      l.user_name.toLowerCase().includes(search.toLowerCase()) ||
      l.details.toLowerCase().includes(search.toLowerCase()) ||
      (l.mission_name ?? "").toLowerCase().includes(search.toLowerCase());
    const matchAction = actionFilter === "all" || l.action === actionFilter;
    const matchRole = roleFilter === "all" || l.user_role === roleFilter;
    return matchSearch && matchAction && matchRole;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Piste d'audit</h1>
          <p className="mt-0.5 text-sm text-muted-foreground">
            {logs.length} événement{logs.length !== 1 ? "s" : ""} enregistré{logs.length !== 1 ? "s" : ""}
          </p>
        </div>
        <Button
          variant="outline"
          className="gap-2 self-start"
          onClick={() => exportCSV(filtered)}
          disabled={filtered.length === 0}
        >
          <Download className="h-4 w-4" />
          Exporter CSV
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Rechercher par utilisateur, mission, détails…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select
          value={actionFilter}
          onValueChange={(v) => setActionFilter(v as AuditLogAction | "all")}
        >
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Toutes les actions" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Toutes les actions</SelectItem>
            {ALL_ACTIONS.map((a) => (
              <SelectItem key={a} value={a}>
                {AUDIT_ACTION_LABELS_FR[a]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={roleFilter}
          onValueChange={(v) => setRoleFilter(v as UserRole | "all")}
        >
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Tous les rôles" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Tous les rôles</SelectItem>
            {(["auditor", "manager", "partner", "admin"] as UserRole[]).map((r) => (
              <SelectItem key={r} value={r}>
                {ROLE_LABELS_FR[r]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {(search || actionFilter !== "all" || roleFilter !== "all") && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSearch("");
              setActionFilter("all");
              setRoleFilter("all");
            }}
          >
            Réinitialiser
          </Button>
        )}
        <span className="ml-auto text-sm text-muted-foreground">
          {filtered.length} résultat{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Views */}
      <Tabs defaultValue="timeline">
        <TabsList>
          <TabsTrigger value="timeline" className="gap-1.5">
            <GitBranch className="h-3.5 w-3.5" />
            Chronologie
          </TabsTrigger>
          <TabsTrigger value="table" className="gap-1.5">
            <List className="h-3.5 w-3.5" />
            Tableau
          </TabsTrigger>
        </TabsList>

        {/* ── Timeline View ─────────────────────── */}
        <TabsContent value="timeline" className="mt-4">
          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full rounded-lg" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="relative space-y-1">
              {/* Vertical line */}
              <div className="absolute left-[18px] top-3 h-full w-px bg-border" />

              {filtered.map((log, idx) => {
                const colorClass = ACTION_ICON_COLOR[log.action] ?? "bg-gray-100 text-gray-600";
                const isFirst = idx === 0;

                return (
                  <div key={log.id} className="relative flex gap-4 pl-10">
                    {/* Dot */}
                    <div
                      className={`absolute left-0 flex h-9 w-9 items-center justify-center rounded-full border-2 border-white shadow-sm ${colorClass}`}
                    >
                      <Shield className="h-3.5 w-3.5" />
                    </div>

                    <Card className={`mb-2 flex-1 ${isFirst ? "border-pwc-orange/30" : ""}`}>
                      <CardContent className="py-3">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div className="space-y-0.5">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-medium text-sm">{log.user_name}</span>
                              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                                {ROLE_LABELS_FR[log.user_role]}
                              </Badge>
                              <span className="text-xs font-medium text-pwc-orange">
                                {AUDIT_ACTION_LABELS_FR[log.action]}
                              </span>
                            </div>
                            <p className="text-xs text-muted-foreground">{log.details}</p>
                            {log.mission_name && (
                              <p className="text-[10px] text-muted-foreground/70">
                                Mission : {log.mission_name}
                              </p>
                            )}
                          </div>
                          <span className="flex items-center gap-1 text-[10px] text-muted-foreground whitespace-nowrap shrink-0">
                            <Clock className="h-3 w-3" />
                            {formatDateTime(log.timestamp)}
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                );
              })}
            </div>
          )}
        </TabsContent>

        {/* ── Table View ──────────────────────────── */}
        <TabsContent value="table" className="mt-4">
          {isLoading ? (
            <Skeleton className="h-64 w-full rounded-lg" />
          ) : filtered.length === 0 ? (
            <EmptyState />
          ) : (
            <Card>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-muted/30">
                      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Date</th>
                      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Utilisateur</th>
                      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Rôle</th>
                      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Action</th>
                      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Mission</th>
                      <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Détails</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {filtered.map((log) => (
                      <tr key={log.id} className="hover:bg-muted/20">
                        <td className="whitespace-nowrap px-4 py-2 text-xs text-muted-foreground">
                          {formatDateTime(log.timestamp)}
                        </td>
                        <td className="px-4 py-2">
                          <div className="flex items-center gap-1.5">
                            <div className="flex h-5 w-5 items-center justify-center rounded-full bg-pwc-orange text-[9px] font-bold text-white">
                              {log.user_name.charAt(0)}
                            </div>
                            <span className="font-medium text-xs">{log.user_name}</span>
                          </div>
                        </td>
                        <td className="px-4 py-2">
                          <Badge variant="outline" className="text-[10px]">
                            {ROLE_LABELS_FR[log.user_role]}
                          </Badge>
                        </td>
                        <td className="whitespace-nowrap px-4 py-2">
                          <span
                            className={`rounded px-2 py-0.5 text-[10px] font-medium ${ACTION_ICON_COLOR[log.action] ?? "bg-gray-100 text-gray-600"}`}
                          >
                            {AUDIT_ACTION_LABELS_FR[log.action]}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-xs text-muted-foreground">
                          {log.mission_name ?? "—"}
                        </td>
                        <td className="max-w-[280px] truncate px-4 py-2 text-xs text-muted-foreground">
                          {log.details}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function EmptyState() {
  return (
    <Card>
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <Shield className="h-12 w-12 text-muted-foreground/30" />
        <p className="mt-4 font-medium text-muted-foreground">Aucun événement enregistré</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Les actions des utilisateurs apparaîtront ici.
        </p>
      </CardContent>
    </Card>
  );
}
