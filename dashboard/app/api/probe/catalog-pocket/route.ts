import { NextResponse } from "next/server";
import {
  ensureCatalogPocketProbe,
  buildCatalogPocketDigest,
} from "@/lib/catalogPocketProbe";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    ensureCatalogPocketProbe();
    const digest = buildCatalogPocketDigest();
    return NextResponse.json(digest);
  } catch (err) {
    console.error("[API /probe/catalog-pocket]", err);
    return NextResponse.json(
      { error: "catalog_pocket_digest_failed", detail: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  }
}
