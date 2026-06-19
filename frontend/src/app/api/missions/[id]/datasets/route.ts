import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import path from "path";
import { datasetRepository } from "@/lib/db/repositories/datasetRepository";
import { auditLogRepository } from "@/lib/db/repositories/auditLogRepository";
import { missionRepository } from "@/lib/db/repositories/missionRepository";
import type { DatasetCategory, UserRole } from "@/types";
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
  const { id } = await params;
  const datasets = await datasetRepository.getByMission(id);
  return NextResponse.json(datasets);
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

  const caller = await getCaller();
  if (!caller) return NextResponse.json({ error: "Non authentifié." }, { status: 401 });

  const datasetId = uuidv4();

  // Persist file to disk so it can be re-used for future analyses
  let storagePath: string | undefined;
  try {
    const uploadDir = path.join(process.cwd(), "uploads", id, datasetId);
    fs.mkdirSync(uploadDir, { recursive: true });
    const filePath = path.join(uploadDir, file.name);
    const bytes = await file.arrayBuffer();
    fs.writeFileSync(filePath, Buffer.from(bytes));
    storagePath = filePath;
  } catch {
    // Non-blocking — metadata still saved even if disk write fails
  }

  const dataset = await datasetRepository.add({
    id: datasetId,
    mission_id: id,
    uploaded_by_id: caller.id,
    name: file.name,
    original_name: file.name,
    category: category || "transactions",
    file_size: file.size,
    file_type: file.type || "text/csv",
    status: "uploaded",
    storage_path: storagePath,
  });

  const mission = await missionRepository.getById(id);
  await auditLogRepository.add({
    action: "dataset.upload",
    user_id: caller.id,
    user_name: caller.name,
    user_role: caller.role,
    mission_id: id,
    mission_name: mission?.name,
    details: `Dataset uploadé : "${file.name}" (${(file.size / 1024).toFixed(0)} KB) — Catégorie : ${category}`,
  });

  return NextResponse.json(dataset, { status: 201 });
}
