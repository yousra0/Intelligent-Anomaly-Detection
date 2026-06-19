import { NextResponse, type NextRequest } from "next/server";

const PUBLIC_PATHS = ["/login", "/api/auth/login"];
const STATIC_EXT = /\.(?:png|jpg|jpeg|gif|webp|svg|ico|woff2?|ttf|eot|css|js)$/i;

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public paths, Next.js internals, and static files from /public
  if (
    PUBLIC_PATHS.some((p) => pathname.startsWith(p)) ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/favicon") ||
    STATIC_EXT.test(pathname)
  ) {
    return NextResponse.next();
  }

  // Check for JWT cookie (set by the login API route)
  const token = request.cookies.get("pwc_token")?.value;
  if (!token && !pathname.startsWith("/api")) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("from", pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
