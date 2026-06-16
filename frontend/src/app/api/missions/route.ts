import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { missionStore } from "@/lib/store/missionStore";
import { auditLogStore } from "@/lib/store/auditLogStore";
import { userStore } from "@/lib/store/userStore";
import type { CreateMissionPayload, UserRole } from "@/types";
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

export async function GET() {
  return NextResponse.json(missionStore.getAll());
}

export async function POST(req: Request) {
  const body = (await req.json()) as CreateMissionPayload;
  const caller = await getCallerFromCookie();

  const mission = missionStore.add({
    id: uuidv4(),
    ...body,
    status: "active",
    created_by: caller?.id ?? "u1",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  });

  // Resolve assigned auditor name for audit log
  const assignedUser = body.assigned_to ? userStore.getById(body.assigned_to) : undefined;

  auditLogStore.add({
    action: "mission.create",
    user_id: caller?.id ?? "u1",
    user_name: caller?.name ?? "Inconnu",
    user_role: caller?.role ?? "auditor",
    mission_id: mission.id,
    mission_name: mission.name,
    details: `Mission créée : "${mission.name}" — ${mission.company_name}${assignedUser ? ` — Assignée à ${assignedUser.name}` : ""}`,
  });

  return NextResponse.json(mission, { status: 201 });
}
