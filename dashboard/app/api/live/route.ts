import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/** Liveness mínimo — sem probes, sem disco. Em produção com `node server.js`, `/api/live` é atendido antes do App Router. */
export async function GET() {
  console.log("[live] hit GET (App Router)");
  const body = { ok: true, live: true, ts: new Date().toISOString() };
  console.log("[live] response sent GET (App Router)");
  return NextResponse.json(body);
}

export async function HEAD() {
  console.log("[live] hit HEAD (App Router)");
  const res = new NextResponse(null, { status: 200 });
  console.log("[live] response sent HEAD (App Router)");
  return res;
}
