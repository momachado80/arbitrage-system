const path = require("path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    instrumentationHook: true,
  },
  webpack: (config, { isServer, nextRuntime }) => {
    if (isServer && nextRuntime === "edge") {
      const real = path.resolve(__dirname, "lib/nodeInstrumentationBootstrap.ts");
      const stub = path.resolve(__dirname, "lib/nodeInstrumentationBootstrap.edge-stub.ts");
      config.resolve.alias = { ...config.resolve.alias, [real]: stub };
    }
    return config;
  },
};

module.exports = nextConfig;
