import { NextResponse } from "next/server";
import { SignJWT } from "jose";
import { userRepository } from "@/lib/db/repositories/userRepository";
import type { LoginCredentials, AuthResponse } from "@/types";

const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET ?? "change-me-in-production-at-least-32-chars!!"
);

export async function POST(req: Request) {
  const body = (await req.json()) as LoginCredentials;

  if (!body.email || !body.password) {
    return NextResponse.json({ error: "Email et mot de passe requis." }, { status: 400 });
  }

  if (!body.email.endsWith("@pwc.com")) {
    return NextResponse.json(
      { error: "Seules les adresses @pwc.com sont autorisées." },
      { status: 403 }
    );
  }

  const userRecord = await userRepository.getByEmail(body.email);
  if (!userRecord) {
    return NextResponse.json({ error: "Identifiants invalides." }, { status: 401 });
  }

  const passwordValid = await userRepository.verifyPassword(userRecord.password, body.password);
  if (!passwordValid) {
    return NextResponse.json({ error: "Identifiants invalides." }, { status: 401 });
  }

  const userPublic = await userRepository.getById(userRecord.id);
  if (!userPublic) {
    return NextResponse.json({ error: "Utilisateur introuvable." }, { status: 404 });
  }

  const expiresIn = parseInt(process.env.JWT_EXPIRY ?? "28800");

  const token = await new SignJWT({
    sub: userRecord.id,
    email: userRecord.email,
    name: userRecord.name,
    role: userRecord.role,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(Math.floor(Date.now() / 1000) + expiresIn)
    .sign(JWT_SECRET);

  const response = NextResponse.json<AuthResponse>({ user: userPublic, token });

  response.cookies.set("pwc_token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: expiresIn,
    path: "/",
  });

  // Audit login event (fire-and-forget — do not block response)
  import("@/lib/db/repositories/auditLogRepository").then(({ auditLogRepository }) => {
    auditLogRepository.add({
      action: "login",
      user_id: userRecord.id,
      user_name: userRecord.name,
      user_role: userRecord.role as import("@/types").UserRole,
      details: "Connexion réussie",
    }).catch(() => {});
  }).catch(() => {});

  return response;
}
