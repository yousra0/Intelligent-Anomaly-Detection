import { NextResponse } from "next/server";
import { userRepository } from "@/lib/db/repositories/userRepository";
import { auditLogRepository } from "@/lib/db/repositories/auditLogRepository";
import type { UserRole, UserStatus } from "@/types";
import { jwtVerify } from "jose";
import { cookies } from "next/headers";

const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET ?? "change-me-in-production-at-least-32-chars!!"
);

async function getCaller() {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("pwc_token")?.value;
    if (!token) return null;
    const { payload } = await jwtVerify(token, JWT_SECRET);
    return { id: payload.sub as string, name: payload.name as string, role: payload.role as UserRole };
  } catch {
    return null;
  }
}

interface Params { params: Promise<{ id: string }> }

export async function GET(_req: Request, { params }: Params) {
  const { id } = await params;
  const user = await userRepository.getById(id);
  if (!user) return NextResponse.json({ error: "Utilisateur introuvable." }, { status: 404 });
  return NextResponse.json(user);
}

export async function PATCH(req: Request, { params }: Params) {
  const { id } = await params;
  const caller = await getCaller();

  try {
    const body = await req.json();

    if (body.action === "setStatus") {
      const user = await userRepository.setStatus(id, body.status as "active" | "inactive" | "suspended");
      if (!user) return NextResponse.json({ error: "Utilisateur introuvable." }, { status: 404 });

      await auditLogRepository.add({
        action: body.status === "active" ? "user_activate" : "user_disable",
        user_id: caller?.id ?? "unknown",
        user_name: caller?.name ?? "Inconnu",
        user_role: caller?.role ?? "admin",
        details: `Statut utilisateur modifié : ${user.name} → ${body.status}`,
      });

      return NextResponse.json(user);
    }

    if (body.action === "resetPassword") {
      if (!body.password) {
        return NextResponse.json({ error: "Mot de passe requis." }, { status: 400 });
      }
      await userRepository.resetPassword(id, body.password);

      await auditLogRepository.add({
        action: "user_reset_password",
        user_id: caller?.id ?? "unknown",
        user_name: caller?.name ?? "Inconnu",
        user_role: caller?.role ?? "admin",
        details: `Mot de passe réinitialisé pour l'utilisateur ID ${id}`,
      });

      return NextResponse.json({ success: true });
    }

    // General update
    const { first_name, last_name, ...rest } = body;
    const patch: Parameters<typeof userRepository.update>[1] = { ...rest };
    if (first_name || last_name) {
      const current = await userRepository.getById(id);
      if (!current) return NextResponse.json({ error: "Utilisateur introuvable." }, { status: 404 });
      const nameParts = current.name.split(" ");
      patch.name = `${first_name ?? nameParts[0]} ${last_name ?? nameParts.slice(1).join(" ")}`.trim();
    }
    const user = await userRepository.update(id, patch);
    if (!user) return NextResponse.json({ error: "Utilisateur introuvable." }, { status: 404 });

    await auditLogRepository.add({
      action: "user_update",
      user_id: caller?.id ?? "unknown",
      user_name: caller?.name ?? "Inconnu",
      user_role: caller?.role ?? "admin",
      details: `Profil utilisateur mis à jour : ${user.name}`,
    });

    return NextResponse.json(user);
  } catch {
    return NextResponse.json({ error: "Requête invalide." }, { status: 400 });
  }
}
