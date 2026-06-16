import { NextResponse } from "next/server";
import { analysisRunStore } from "@/lib/store/analysisRunStore";
import { auditLogStore } from "@/lib/store/auditLogStore";
import type { AnalysisRunRecord, UserRole } from "@/types";
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

  const runs = missionId
    ? analysisRunStore.getByMission(missionId)
    : analysisRunStore.getAll();

  return NextResponse.json(runs);
}

export async function POST(req: Request) {
  const caller = await getCallerFromCookie();
  const body = (await req.json()) as Omit<AnalysisRunRecord, "user_id" | "user_name">;

  const run = analysisRunStore.add({
    ...body,
    user_id: caller?.id ?? "u1",
    user_name: caller?.name ?? "Inconnu",
  });

  // Log to audit trail
  auditLogStore.add({
    action: run.status === "completed" ? "analysis.complete" : "analysis.start",
    user_id: caller?.id ?? "u1",
    user_name: caller?.name ?? "Inconnu",
    user_role: caller?.role ?? "auditor",
    mission_id: run.mission_id,
    mission_name: run.mission_name,
    details: `Analyse ${run.status === "completed" ? "terminée" : "démarrée"} — ${run.dataset_name} — Modèle : ${run.model_mode}${run.result ? ` — ${run.result.n_fraud} anomalie(s) détectée(s)` : ""}`,
  });

  return NextResponse.json(run, { status: 201 });
}
