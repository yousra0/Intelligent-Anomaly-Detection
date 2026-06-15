import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/ml/:path*",
        destination: `${process.env.FASTAPI_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
  images: {
    domains: ["localhost"],
  },
  // Allow large CSV/XLSX uploads up to 500 MB through Next.js API routes
  experimental: {
    serverActions: {
      bodySizeLimit: "500mb",
    },
  },
};

export default nextConfig;
