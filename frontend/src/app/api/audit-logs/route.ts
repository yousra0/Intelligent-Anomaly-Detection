import { NextResponse } from "next/server";
import { auditLogStore } from "@/lib/store/auditLogStore";
import type { AuditLogAction, UserRole } from "@/types";
import { jwtVerify } from "jose";
import { cookies } from "next/headers";

const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET ?? "change-me-in-production-at-least-32-chars!!"
);

async function getCallerFromCookie() {
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

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const missionId = searchParams.get("mission_id");
  const limit = parseInt(searchParams.get("limit") ?? "200");

  const logs = missionId
    ? auditLogStore.getByMission(missionId)
    : auditLogStore.getAll();

  return NextResponse.json(logs.slice(0, limit));
}

export async function POST(req: Request) {
  const caller = await getCallerFromCookie();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });

  const body = (await req.json()) as {
    action: AuditLogAction;
    mission_id?: string;
    mission_name?: string;
    details: string;
  };

  const log = auditLogStore.add({
    action: body.action,
    user_id: caller.id,
    user_name: caller.name,
    user_role: caller.role,
    mission_id: body.mission_id,
    mission_name: body.mission_name,
    details: body.details,
  });

  return NextResponse.json(log, { status: 201 });
}
