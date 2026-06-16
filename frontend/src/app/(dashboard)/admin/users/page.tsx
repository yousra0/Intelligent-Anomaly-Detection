"use client";

import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { userService } from "@/lib/api/userService";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Users, ShieldCheck, Mail } from "lucide-react";
import { usePermissions } from "@/lib/hooks/usePermissions";
import { ROLE_LABELS_FR } from "@/lib/utils";
import type { UserRole } from "@/types";

const ROLE_BADGE_VARIANT: Record<UserRole, string> = {
  admin:   "bg-red-100 text-red-700",
  manager: "bg-blue-100 text-blue-700",
  partner: "bg-purple-100 text-purple-700",
  auditor: "bg-green-100 text-green-700",
};

export default function AdminUsersPage() {
  const { can } = usePermissions();
  const router = useRouter();

  // Redirect non-admins
  useEffect(() => {
    if (!can("admin.users")) {
      router.replace("/missions");
    }
  }, [can, router]);

  const { data: users = [], isLoading } = useQuery({
    queryKey: ["users", "all"],
    queryFn: userService.getAll,
    enabled: can("admin.users"),
  });

  if (!can("admin.users")) return null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Gestion des utilisateurs</h1>
        <p className="mt-0.5 text-sm text-muted-foreground">
          Administration · {users.length} utilisateur{users.length !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Role distribution */}
      <div className="grid gap-4 sm:grid-cols-4">
        {(["admin", "manager", "partner", "auditor"] as UserRole[]).map((role) => (
          <Card key={role}>
            <CardContent className="pt-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-muted p-2">
                  <ShieldCheck className="h-4 w-4 text-pwc-orange" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">{ROLE_LABELS_FR[role]}</p>
                  <p className="text-xl font-bold">
                    {isLoading ? "—" : users.filter((u) => u.role === role).length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* User list */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Users className="h-4 w-4 text-pwc-orange" />
            Liste des utilisateurs
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-14 rounded-lg" />
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/30">
                    <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Utilisateur</th>
                    <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Email</th>
                    <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Rôle</th>
                    <th className="px-4 py-2.5 text-left text-xs font-semibold text-muted-foreground">Statut</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {users.map((user) => (
                    <tr key={user.id} className="hover:bg-muted/20">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-pwc-orange text-xs font-bold text-white">
                            {user.name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase()}
                          </div>
                          <span className="font-medium">{user.name}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <Mail className="h-3 w-3" />
                          {user.email}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                            ROLE_BADGE_VARIANT[user.role as UserRole] ?? "bg-gray-100 text-gray-700"
                          }`}
                        >
                          {ROLE_LABELS_FR[user.role as UserRole]}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-1.5 text-xs text-green-700">
                          <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                          Actif
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <p className="mt-4 text-xs text-muted-foreground">
                * La gestion complète des utilisateurs (création, modification, désactivation) nécessite
                l'intégration PostgreSQL. Connectez la base de données et remplacez le store en mémoire
                dans <code className="rounded bg-muted px-1">src/lib/store/userStore.ts</code>.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
