"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalyticsData } from "@/lib/api";
import { fetchAnalytics } from "@/lib/api";
import { fetchLiveAnalytics } from "@/lib/liveApi";
import Header from "@/components/Header";
import MarketRegimeCard from "@/components/MarketRegimeCard";
import OpportunitiesTable from "@/components/OpportunitiesTable";
import ArbitrageTable from "@/components/ArbitrageTable";
import ShadowTradingPanel from "@/components/ShadowTradingPanel";
import RiskPanel from "@/components/RiskPanel";

const REFRESH_MS = 5_000;

export default function DashboardPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const liveData = await fetchLiveAnalytics();
      setData(liveData);
      setError(null);
      setLastUpdate(new Date());
    } catch {
      try {
        const backendData = await fetchAnalytics();
        setData(backendData);
        setError(backendData.error ?? null);
        setLastUpdate(new Date());
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Erro de conexão");
      }
    }
  }, []);

  useEffect(() => {
    refresh();
    timerRef.current = setInterval(refresh, REFRESH_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [refresh]);

  return (
    <div className="max-w-[1440px] mx-auto p-3 sm:p-4 space-y-3">
      <Header data={data} lastUpdate={lastUpdate} error={error} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="lg:col-span-1">
          <MarketRegimeCard data={data} />
        </div>
        <div className="lg:col-span-2">
          <OpportunitiesTable data={data} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ArbitrageTable data={data} />
        <ShadowTradingPanel data={data} />
      </div>

      <RiskPanel data={data} />

      <footer className="text-center text-terminal-muted text-[10px] py-2">
        POLYMARKET ALPHA TERMINAL — Atualização a cada 5 segundos
      </footer>
    </div>
  );
}
