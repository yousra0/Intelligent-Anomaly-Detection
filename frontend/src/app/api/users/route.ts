import { NextResponse } from "next/server";
import { userRepository } from "@/lib/db/repositories/userRepository";
import { auditLogRepository } from "@/lib/db/repositories/auditLogRepository";
import type { CreateUserPayload, UserRole } from "@/types";
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

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const role = searchParams.get("role");

  const users = role === "auditor"
    ? await userRepository.getAuditors()
    : await userRepository.getAll();

  return NextResponse.json(users, {
    headers: { "Cache-Control": "private, max-age=60, stale-while-revalidate=300" },
  });
}

export async function POST(req: Request) {
  const caller = await getCaller();
  if (!caller || caller.role !== "admin") {
    return NextResponse.json({ error: "Accès refusé." }, { status: 403 });
  }

  try {
    const body = (await req.json()) as CreateUserPayload;
    if (!body.first_name || !body.last_name || !body.email || !body.password) {
      return NextResponse.json({ error: "Champs requis manquants." }, { status: 400 });
    }
    const existing = await userRepository.getByEmail(body.email);
    if (existing) {
      return NextResponse.json({ error: "Cet email est déjà utilisé." }, { status: 409 });
    }
    const user = await userRepository.create(body);

    await auditLogRepository.add({
      action: "user_create",
      user_id: caller.id,
      user_name: caller.name,
      user_role: caller.role,
      details: `Utilisateur créé : ${user.name} (${user.email}) — Rôle : ${user.role}`,
    });

    return NextResponse.json(user, { status: 201 });
  } catch {
    return NextResponse.json({ error: "Requête invalide." }, { status: 400 });
  }
}
