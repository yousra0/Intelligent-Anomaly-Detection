"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/AuthContext";
import { usePermissions } from "@/lib/hooks/usePermissions";
import { ROLE_LABELS_FR } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import {
  LayoutDashboard,
  FolderOpen,
  History,
  FileText,
  Users,
  Settings,
  ChevronDown,
  LogOut,
  User,
  Bell,
  Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: typeof LayoutDashboard;
}

export function Navbar() {
  const { user, logout } = useAuth();
  const { can, role } = usePermissions();
  const pathname = usePathname();
  const router = useRouter();

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  // Build navigation items based on role
  const navItems: NavItem[] = [
    { href: "/dashboard", label: "Tableau de bord", icon: LayoutDashboard },
    { href: "/missions", label: "Missions", icon: FolderOpen },
    { href: "/history", label: "Historique", icon: History },
    { href: "/reports", label: "Rapports", icon: FileText },
  ];

  const managerItems: NavItem[] = can("mission.assign")
    ? [{ href: "/missions", label: "Gestion missions", icon: Shield }]
    : [];

  const adminItems: NavItem[] = can("admin.users")
    ? [
        { href: "/admin/users", label: "Utilisateurs", icon: Users },
        { href: "/admin/settings", label: "Paramètres", icon: Settings },
      ]
    : [];

  const allItems = [...navItems, ...managerItems, ...adminItems];

  // Deduplicate by href
  const uniqueItems = allItems.filter(
    (item, idx, arr) => arr.findIndex((x) => x.href === item.href) === idx
  );

  const userInitials = user?.name
    ? user.name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase()
    : "?";

  const roleLabel = role ? ROLE_LABELS_FR[role] : "";

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-white shadow-sm">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* ── Left: Logo + nav ─────────────────────── */}
        <div className="flex items-center gap-6">
          {/* PwC wordmark */}
          <Link href="/dashboard" className="flex items-center gap-2 shrink-0">
            <div className="flex items-center justify-center rounded bg-pwc-orange px-2 py-1">
              <span className="text-xs font-black tracking-tight text-white">PwC</span>
            </div>
            <span className="hidden text-sm font-semibold text-gray-700 sm:block">
              Audit Analytics
            </span>
          </Link>

          {/* Navigation links */}
          <nav className="hidden md:flex items-center gap-1">
            {uniqueItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === "/missions"
                  ? pathname.startsWith("/missions")
                  : pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-pwc-orange/10 text-pwc-orange"
                      : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                  )}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* ── Right: Audit trail + notifications + user ── */}
        <div className="flex items-center gap-2">
          {/* Audit trail shortcut */}
          <Link
            href="/audit-trail"
            className={cn(
              "hidden sm:flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
              pathname === "/audit-trail"
                ? "bg-pwc-orange/10 text-pwc-orange"
                : "text-gray-600 hover:bg-gray-100"
            )}
          >
            <Shield className="h-3.5 w-3.5" />
            <span className="hidden lg:block">Piste d'audit</span>
          </Link>

          {/* Notifications bell (placeholder) */}
          <Button variant="ghost" size="icon" className="relative h-8 w-8">
            <Bell className="h-4 w-4 text-gray-500" />
          </Button>

          {/* User dropdown */}
          {user && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="h-9 gap-2 px-2">
                  <div className="flex h-7 w-7 items-center justify-center rounded-full bg-pwc-orange text-white text-xs font-bold shrink-0">
                    {userInitials}
                  </div>
                  <div className="hidden flex-col items-start sm:flex">
                    <span className="text-xs font-semibold leading-tight">{user.name}</span>
                    <span className="text-[10px] text-muted-foreground leading-tight">{roleLabel}</span>
                  </div>
                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel className="font-normal">
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-semibold">{user.name}</p>
                    <p className="text-xs text-muted-foreground">{user.email}</p>
                    <span className="mt-0.5 inline-block w-fit rounded-full bg-pwc-orange/10 px-2 py-0.5 text-[10px] font-medium text-pwc-orange">
                      {roleLabel}
                    </span>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/missions")}>
                  <FolderOpen className="mr-2 h-4 w-4" />
                  Mes missions
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/history")}>
                  <History className="mr-2 h-4 w-4" />
                  Historique d'analyses
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/audit-trail")}>
                  <Shield className="mr-2 h-4 w-4" />
                  Piste d'audit
                </DropdownMenuItem>
                {can("admin.users") && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/admin/users")}>
                      <Users className="mr-2 h-4 w-4" />
                      Gestion utilisateurs
                    </DropdownMenuItem>
                  </>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="cursor-pointer text-red-600 focus:text-red-600"
                  onClick={handleLogout}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Se déconnecter
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>

      {/* Mobile nav */}
      <nav className="flex items-center gap-1 overflow-x-auto border-t border-border px-4 py-1 md:hidden">
        {uniqueItems.slice(0, 4).map((item) => {
          const Icon = item.icon;
          const isActive =
            item.href === "/missions"
              ? pathname.startsWith("/missions")
              : pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex shrink-0 items-center gap-1 rounded px-2.5 py-1 text-xs font-medium transition-colors",
                isActive ? "bg-pwc-orange/10 text-pwc-orange" : "text-gray-600 hover:bg-gray-100"
              )}
            >
              <Icon className="h-3 w-3" />
              {item.label}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
