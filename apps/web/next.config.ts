import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  transpilePackages: ["@bos/api-client", "@bos/card-schema"],
  poweredByHeader: false,
};

export default nextConfig;
