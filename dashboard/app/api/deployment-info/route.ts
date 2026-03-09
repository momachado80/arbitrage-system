/**
 * Deployment info — exposes public URL when running on Railway.
 * Used by scripts to discover or verify the production dashboard URL.
 * No business logic.
 */

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const domain = process.env.RAILWAY_PUBLIC_DOMAIN;
  const url = domain ? `https://${domain}` : null;
  return NextResponse.json({
    success: true,
    dashboardUrl: url,
    hasRailwayDomain: !!domain,
    timestamp: new Date().toISOString(),
  });
}
