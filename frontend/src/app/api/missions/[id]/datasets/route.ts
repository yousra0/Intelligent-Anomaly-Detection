import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { datasetStore } from "@/lib/store/datasetStore";
import { auditLogStore } from "@/lib/store/auditLogStore";
import { missionStore } from "@/lib/store/missionStore";
import type { DatasetCategory, UserRole } from "@/types";
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
    return { id: payload.sub as string, name: payload.name as string, role: payload.role as UserRole };
  } catch {
    return null;
  }
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  return NextResponse.json(datasetStore.getByMission(id));
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const formData = await req.formData();
  const file = formData.get("file") as File | null;
  const category = formData.get("category") as DatasetCategory;

  if (!file) return NextResponse.json({ error: "Fichier manquant." }, { status: 400 });

  const dataset = datasetStore.add({
    id: uuidv4(),
    mission_id: id,
    name: file.name,
    category: category || "transactions",
    file_size: file.size,
    file_type: file.type || "text/csv",
    status: "uploaded",
    uploaded_at: new Date().toISOString(),
  });

  const caller = await getCallerFromCookie();
  const mission = missionStore.getById(id);
  auditLogStore.add({
    action: "dataset.upload",
    user_id: caller?.id ?? "u1",
    user_name: caller?.name ?? "Inconnu",
    user_role: caller?.role ?? "auditor",
    mission_id: id,
    mission_name: mission?.name,
    details: `Dataset uploadé : "${file.name}" (${(file.size / 1024).toFixed(0)} KB) — Catégorie : ${category}`,
  });

  return NextResponse.json(dataset, { status: 201 });
}
