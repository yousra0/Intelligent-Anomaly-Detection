import { NextResponse } from "next/server";
import { jwtVerify } from "jose";
import { cookies } from "next/headers";

const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET ?? "change-me-in-production-at-least-32-chars!!"
);

export async function GET() {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("pwc_token")?.value;

    if (!token) {
      return NextResponse.json({ error: "Non authentifié" }, { status: 401 });
    }

    const { payload } = await jwtVerify(token, JWT_SECRET);

    return NextResponse.json({
      id: payload.sub as string,
      email: payload.email as string,
      name: payload.name as string,
      role: payload.role as string,
    });
  } catch {
    return NextResponse.json({ error: "Token invalide" }, { status: 401 });
  }
}
