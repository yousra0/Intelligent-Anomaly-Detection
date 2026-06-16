"use client";

import { useAuth } from "@/lib/auth/AuthContext";
import { useRouter } from "next/navigation";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { ChevronDown, LogOut, User, Building2 } from "lucide-react";

interface TopBarProps {
  title?: string;
}

export function TopBar({ title }: TopBarProps) {
  const { user, logout } = useAuth();
  const router = useRouter();

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <header className="sticky top-0 z-40 h-14 border-b border-border bg-white shadow-sm">
      <div className="flex h-full items-center justify-between px-6">
        {/* Left: Logo + breadcrumb */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            {/* PwC wordmark */}
            <div className="flex items-center justify-center rounded bg-pwc-orange px-2 py-1">
              <span className="text-xs font-black tracking-tight text-white">PwC</span>
            </div>
            <span className="text-sm font-medium text-muted-foreground hidden sm:block">
              Audit Analytics Platform
            </span>
          </div>
          {title && (
            <>
              <span className="text-muted-foreground/40">/</span>
              <span className="text-sm font-semibold text-foreground">{title}</span>
            </>
          )}
        </div>

        {/* Right: User menu */}
        {user && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-9 gap-2 px-3">
                <div className="flex h-7 w-7 items-center justify-center rounded-full bg-pwc-orange text-white text-xs font-bold">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div className="hidden flex-col items-start sm:flex">
                  <span className="text-xs font-semibold">{user.name}</span>
                  <span className="text-xs text-muted-foreground capitalize">{user.role.replace("_", " ")}</span>
                </div>
                <ChevronDown className="h-3 w-3 text-muted-foreground" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52">
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium">{user.name}</p>
                  <p className="text-xs text-muted-foreground">{user.email}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="cursor-pointer">
                <User className="mr-2 h-4 w-4" />
                Mon profil
              </DropdownMenuItem>
              <DropdownMenuItem
                className="cursor-pointer"
                onClick={() => router.push("/missions")}
              >
                <Building2 className="mr-2 h-4 w-4" />
                Mes missions
              </DropdownMenuItem>
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
    </header>
  );
}
