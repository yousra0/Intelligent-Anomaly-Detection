import { NextResponse } from "next/server";
import fs from "fs";
import { datasetRepository } from "@/lib/db/repositories/datasetRepository";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string; datasetId: string }> }
) {
  const { id: missionId, datasetId } = await params;

  const meta = await datasetRepository.getStoragePath(datasetId);

  if (!meta?.storagePath) {
    return NextResponse.json(
      { error: "Fichier non disponible — réimportez ce dataset." },
      { status: 404 }
    );
  }

  if (!fs.existsSync(meta.storagePath)) {
    return NextResponse.json(
      { error: "Fichier introuvable sur le serveur — réimportez ce dataset." },
      { status: 404 }
    );
  }

  const bytes = fs.readFileSync(meta.storagePath);
  const fileName = encodeURIComponent(meta.originalName);

  return new Response(bytes, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename*=UTF-8''${fileName}`,
      "Content-Length": String(bytes.byteLength),
    },
  });
}
