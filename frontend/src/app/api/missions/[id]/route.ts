import { NextResponse } from "next/server";
import { missionStore } from "@/lib/store/missionStore";
import { auditLogStore } from "@/lib/store/auditLogStore";
import type { UserRole } from "@/types";
import { jwtVerify } from "jose";
import { cookies } from "next/headers";

const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET ?? "change-me-in-production-at-least-32-chars!!"
);

async function getCallerFromCookie(): Promise<{ id: string; name: string; role: UserRole } | null> {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("pwc_token")?.value;
    if (!token) return null;
    const { payload } = await jwtVerify(token, JWT_SECRET);
    return {
      id: payload.sub as string,
      name: payload.name as string,
      role: payload.role as UserRole,
    };
  } catch {
    return null;
  }
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const mission = missionStore.getById(id);
  if (!mission) return NextResponse.json({ error: "Mission introuvable." }, { status: 404 });
  return NextResponse.json(mission);
}

export async function PUT(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await req.json();
  const mission = missionStore.update(id, body);
  if (!mission) return NextResponse.json({ error: "Mission introuvable." }, { status: 404 });

  const caller = await getCallerFromCookie();
  auditLogStore.add({
    action: "mission.update",
    user_id: caller?.id ?? "u1",
    user_name: caller?.name ?? "Inconnu",
    user_role: caller?.role ?? "auditor",
    mission_id: id,
    mission_name: mission.name,
    details: `Mission mise à jour : "${mission.name}"`,
  });

  return NextResponse.json(mission);
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const mission = missionStore.getById(id);
  const ok = missionStore.remove(id);
  if (!ok) return NextResponse.json({ error: "Mission introuvable." }, { status: 404 });

  const caller = await getCallerFromCookie();
  auditLogStore.add({
    action: "mission.delete",
    user_id: caller?.id ?? "u1",
    user_name: caller?.name ?? "Inconnu",
    user_role: caller?.role ?? "auditor",
    mission_id: id,
    mission_name: mission?.name,
    details: `Mission supprimée : "${mission?.name ?? id}"`,
  });

  return NextResponse.json({ ok: true });
}
