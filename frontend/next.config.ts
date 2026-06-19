import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  compress: true,
  poweredByHeader: false,
  // Required for Docker standalone deployment
  output: process.env.DOCKER_BUILD === "true" ? "standalone" : undefined,
  outputFileTracingRoot: require("path").join(__dirname, "../"),

  async rewrites() {
    return [
      {
        source: "/ml/:path*",
        destination: `${process.env.FASTAPI_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },

  // Aggressive response caching for static assets
  async headers() {
    return [
      {
        source: "/png_logo.png",
        headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable", }],
      },
      {
        source: "/_next/static/:path*",
        headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }],
      },
    ];
  },

  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
    ],
    // Use modern formats (avif, webp) for automatic compression
    formats: ["image/avif", "image/webp"],
  },

  experimental: {
    // Tree-shake barrel exports — critical for lucide-react and recharts bundle size
    optimizePackageImports: [
      "lucide-react",
      "recharts",
      "@radix-ui/react-dialog",
      "@radix-ui/react-select",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-tabs",
      "@radix-ui/react-progress",
      "@radix-ui/react-separator",
      "@radix-ui/react-label",
      "@radix-ui/react-slot",
      "date-fns",
    ],
    serverActions: {
      bodySizeLimit: "500mb",
    },
  },
};

export default nextConfig;
