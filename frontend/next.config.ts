import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8001/api/:path*",
      },
      {
        source: "/results/:path*",
        destination: "http://127.0.0.1:8001/results/:path*",
      },
    ];
  },
};

export default nextConfig;
