import { NextResponse } from "next/server";
import { missionRepository } from "@/lib/db/repositories/missionRepository";
import { auditLogRepository } from "@/lib/db/repositories/auditLogRepository";
import type { UserRole } from "@/types";
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

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });

  const { id } = await params;
  const mission = await missionRepository.getById(id);
  if (!mission) return NextResponse.json({ error: "Mission introuvable." }, { status: 404 });

  // Auditors can only view missions they are assigned to
  if (
    caller.role === "auditor" &&
    mission.assigned_to !== caller.id &&
    !mission.assigned_auditors?.includes(caller.id)
  ) {
    return NextResponse.json({ error: "Accès refusé." }, { status: 403 });
  }

  return NextResponse.json(mission);
}

export async function PUT(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });
  if (!["manager", "partner", "admin"].includes(caller.role)) {
    return NextResponse.json({ error: "Accès refusé. Rôle requis : manager, partner ou admin." }, { status: 403 });
  }

  const { id } = await params;
  const body = await req.json();
  const mission = await missionRepository.update(id, body);
  if (!mission) return NextResponse.json({ error: "Mission introuvable." }, { status: 404 });

  await auditLogRepository.add({
    action: "mission.update",
    user_id: caller.id,
    user_name: caller.name,
    user_role: caller.role,
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
  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });
  // Only manager and admin can delete (partner cannot per permission matrix)
  if (!["manager", "admin"].includes(caller.role)) {
    return NextResponse.json({ error: "Accès refusé. Rôle requis : manager ou admin." }, { status: 403 });
  }

  const { id } = await params;
  const mission = await missionRepository.getById(id);
  if (!mission) return NextResponse.json({ error: "Mission introuvable." }, { status: 404 });

  await missionRepository.remove(id);

  await auditLogRepository.add({
    action: "mission.delete",
    user_id: caller.id,
    user_name: caller.name,
    user_role: caller.role,
    mission_id: id,
    mission_name: mission.name,
    details: `Mission supprimée : "${mission.name}"`,
  });

  return NextResponse.json({ ok: true });
}
