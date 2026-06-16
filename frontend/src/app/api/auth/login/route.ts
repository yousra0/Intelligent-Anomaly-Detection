import { NextResponse } from "next/server";
import { SignJWT } from "jose";
import { userStore } from "@/lib/store/userStore";
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

  // Authenticate against userStore (credentials from DEMO_PASSWORD env var)
  const userRecord = userStore.getByEmail(body.email);
  if (!userRecord || userRecord.password !== body.password) {
    return NextResponse.json({ error: "Identifiants invalides." }, { status: 401 });
  }

  const { password: _, ...userWithoutPassword } = userRecord;

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

  const response = NextResponse.json<AuthResponse>({
    user: userWithoutPassword,
    token,
  });

  response.cookies.set("pwc_token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: expiresIn,
    path: "/",
  });

  return response;
}
