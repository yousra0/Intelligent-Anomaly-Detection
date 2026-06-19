import { NextResponse } from "next/server";
import { missionRepository } from "@/lib/db/repositories/missionRepository";
import { auditLogRepository } from "@/lib/db/repositories/auditLogRepository";
import { userRepository } from "@/lib/db/repositories/userRepository";
import type { CreateMissionPayload, UserRole } from "@/types";
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
    return {
      id: payload.sub as string,
      name: payload.name as string,
      role: payload.role as UserRole,
    };
  } catch {
    return null;
  }
}

export async function GET() {
  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });

  // Auditors only see missions they are assigned to
  const missions = caller.role === "auditor"
    ? await missionRepository.getByAssignee(caller.id)
    : await missionRepository.getAll();

  return NextResponse.json(missions, {
    headers: { "Cache-Control": "private, max-age=60, stale-while-revalidate=300" },
  });
}

export async function POST(req: Request) {
  const body = (await req.json()) as CreateMissionPayload;
  const caller = await getCaller();

  if (!caller) {
    return NextResponse.json({ error: "Non authentifié." }, { status: 401 });
  }

  if (!["manager", "partner", "admin"].includes(caller.role)) {
    return NextResponse.json({ error: "Accès refusé. Rôle requis : manager, partner ou admin." }, { status: 403 });
  }

  const mission = await missionRepository.create(body, caller.id);

  const assignedUser = body.assigned_to
    ? await userRepository.getById(body.assigned_to)
    : undefined;

  await auditLogRepository.add({
    action: "mission.create",
    user_id: caller.id,
    user_name: caller.name,
    user_role: caller.role,
    mission_id: mission.id,
    mission_name: mission.name,
    details: `Mission créée : "${mission.name}" — ${mission.company_name}${assignedUser ? ` — Assignée à ${assignedUser.name}` : ""}`,
  });

  return NextResponse.json(mission, { status: 201 });
}
