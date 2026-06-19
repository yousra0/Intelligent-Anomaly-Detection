import { NextResponse } from "next/server";
import { analysisRunRepository } from "@/lib/db/repositories/analysisRunRepository";
import { auditLogRepository } from "@/lib/db/repositories/auditLogRepository";
import type { AnalysisRunRecord, UserRole } from "@/types";
import { jwtVerify } from "jose";
import { cookies } from "next/headers";
import { v4 as uuidv4 } from "uuid";

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

export async function GET(req: Request) {
  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const missionId = searchParams.get("mission_id");

  const runs = missionId
    ? await analysisRunRepository.getByMission(missionId)
    : await analysisRunRepository.getAll();

  return NextResponse.json(runs, {
    headers: { "Cache-Control": "private, max-age=60, stale-while-revalidate=300" },
  });
}

export async function POST(req: Request) {
  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });
  // Partners cannot run analyses per permission matrix
  if (caller.role === "partner") {
    return NextResponse.json({ error: "Accès refusé. Les partners ne peuvent pas lancer d'analyses." }, { status: 403 });
  }

  const body = (await req.json()) as Omit<AnalysisRunRecord, "id" | "user_id">;

  const run = await analysisRunRepository.create({
    ...body,
    id: uuidv4(),
    user_id: caller?.id ?? "unknown",
  });

  await auditLogRepository.add({
    action: run.status === "completed" ? "analysis.complete" : "analysis.start",
    user_id: caller?.id ?? "unknown",
    user_name: caller?.name ?? "Inconnu",
    user_role: caller?.role ?? "auditor",
    mission_id: run.mission_id,
    mission_name: run.mission_name,
    details: `Analyse ${run.status === "completed" ? "terminée" : "démarrée"} — ${run.dataset_name} — Modèle : ${run.model_mode}${run.result ? ` — ${(run.result as { n_fraud?: number }).n_fraud ?? 0} anomalie(s) détectée(s)` : ""}`,
  });

  return NextResponse.json(run, { status: 201 });
}
