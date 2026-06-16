import { NextResponse } from "next/server";
import { userStore } from "@/lib/store/userStore";

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const role = searchParams.get("role");

  if (role === "auditor") {
    return NextResponse.json(userStore.getAuditors());
  }

  return NextResponse.json(userStore.getAll());
}
