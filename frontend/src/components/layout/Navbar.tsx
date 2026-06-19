"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth/AuthContext";
import { usePermissions } from "@/lib/hooks/usePermissions";
import { useTheme } from "@/providers/ThemeProvider";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { ROLE_LABELS_FR } from "@/lib/utils";
import { Logo } from "@/components/ui/Logo";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { ChevronDown, LogOut, Sun, Moon, Menu, X, Globe } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { Locale } from "@/lib/i18n/LanguageContext";

export function Navbar() {
  const { user, logout } = useAuth();
  const { can, role } = usePermissions();
  const { theme, toggleTheme } = useTheme();
  const { locale, setLocale, t } = useLanguage();
  const pathname = usePathname();
  const router = useRouter();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  const navItems = [
    { href: "/dashboard", label: t("nav.dashboard") },
    { href: "/missions", label: t("nav.missions") },
    { href: "/history", label: t("nav.history") },
    { href: "/reports", label: t("nav.reports") },
  ];

  if (can("audit.trail")) {
    navItems.push({ href: "/audit-trail", label: t("nav.auditTrail") });
  }
  if (can("admin.users")) {
    navItems.push({ href: "/admin/users", label: t("nav.admin") });
  }

  const userInitials = user?.name
    ? user.name.split(" ").map((n) => n[0]).join("").slice(0, 2).toUpperCase()
    : "?";

  const roleLabel = role
    ? (locale === "fr" ? ROLE_LABELS_FR[role] : t(`role.${role}`))
    : "";

  const isActive = (href: string) =>
    href === "/missions" ? pathname.startsWith("/missions") : pathname === href;

  const otherLocale: Locale = locale === "fr" ? "en" : "fr";

  return (
    <header className="sticky top-0 z-50 border-b bg-[hsl(var(--navbar-bg))] border-[hsl(var(--navbar-border))] shadow-sm">
      <div className="mx-auto flex h-14 max-w-screen-2xl items-center justify-between px-4 sm:px-6 lg:px-8">
        {/* Left: Logo + nav */}
        <div className="flex items-center gap-6 xl:gap-8">
          <Logo href="/dashboard" size="sm" />
          <nav className="hidden md:flex items-center gap-0.5" aria-label="Navigation principale">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors whitespace-nowrap",
                  isActive(item.href)
                    ? "bg-pwc-orange/10 text-pwc-orange"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>

        {/* Right: language + theme + user */}
        <div className="flex items-center gap-0.5">
          {/* Language switcher */}
          <Button
            variant="ghost"
            size="sm"
            className="h-8 gap-1.5 px-2 text-xs font-semibold text-muted-foreground hover:text-foreground"
            onClick={() => setLocale(otherLocale)}
            aria-label={`Switch to ${otherLocale === "fr" ? "French" : "English"}`}
            title={t("common.language")}
          >
            <Globe className="h-3.5 w-3.5" />
            <span className="hidden sm:inline uppercase">{otherLocale}</span>
          </Button>

          {/* Theme toggle */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? t("nav.lightMode") : t("nav.darkMode")}
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4 text-muted-foreground" />
            ) : (
              <Moon className="h-4 w-4 text-muted-foreground" />
            )}
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
                  {t("nav.myMissions")}
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/history")}>
                  {t("nav.analysisHistory")}
                </DropdownMenuItem>
                <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/audit-trail")}>
                  {t("nav.auditLog")}
                </DropdownMenuItem>
                {can("admin.users") && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem className="cursor-pointer" onClick={() => router.push("/admin/users")}>
                      {t("nav.userManagement")}
                    </DropdownMenuItem>
                  </>
                )}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="cursor-pointer text-red-600 focus:text-red-600"
                  onClick={handleLogout}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  {t("nav.logout")}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}

          {/* Mobile menu toggle */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden h-8 w-8"
            onClick={() => setMobileOpen((v) => !v)}
            aria-label="Menu"
          >
            {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* Mobile nav */}
      {mobileOpen && (
        <nav className="md:hidden border-t border-[hsl(var(--navbar-border))] bg-[hsl(var(--navbar-bg))] px-4 py-2 flex flex-col gap-1">
          {navItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive(item.href)
                  ? "bg-pwc-orange/10 text-pwc-orange"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
              onClick={() => setMobileOpen(false)}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
