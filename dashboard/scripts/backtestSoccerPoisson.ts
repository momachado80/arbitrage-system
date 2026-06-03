/**
 * Backtest de calibração — modelo Poisson+ELO vs resultados reais (read-only puro).
 *
 * Lê um CSV de resultados de seleções (Kaggle "international football results":
 * date,home_team,away_team,home_score,away_score,tournament,city,country,neutral),
 * constrói ELO rolling cronológico, e para cada partida (após burn-in) gera as
 * probabilidades do modelo para mercados O/U + BTTS + resultado, comparando com o
 * que de fato aconteceu. Reporta Brier, Brier Skill Score (vs base rate) e curva
 * de calibração por mercado.
 *
 * NÃO testa contra o mercado — testa se o MODELO é calibrado e bate o preditor
 * ingênuo. É o portão #1: se falhar aqui, não há por que comparar com a Polymarket.
 *
 * Sem rede, sem ordens, sem .paper, sem microcapital, sem execução. Só lê arquivo
 * local e imprime estatística.
 *
 * Uso: BACKTEST_RESULTS_CSV=/caminho/results.csv \
 *      [BACKTEST_MIN_YEAR=2010] [BACKTEST_BURN_IN=10] \
 *      ts-node -P tsconfig.worker.json scripts/backtestSoccerPoisson.ts
 */

import fs from "fs";

import { expectedGoalsFromElo, DEFAULT_ELO_PARAMS } from "../lib/soccerEloModel";
import { pOver, pBothTeamsScore, matchResultProbs } from "../lib/poissonGoalsModel";
import { updateElo, DEFAULT_ROLLING_ELO, ELO_INITIAL } from "../lib/rollingElo";
import {
  brierScore,
  brierSkillScore,
  calibrationBins,
  baseRateBrier,
  type ForecastSample,
} from "../lib/forecastCalibration";

const CSV_PATH = process.env.BACKTEST_RESULTS_CSV ?? "";
const MIN_YEAR = parseInt(process.env.BACKTEST_MIN_YEAR ?? "2010", 10) || 2010;
const BURN_IN = parseInt(process.env.BACKTEST_BURN_IN ?? "10", 10) || 10;

interface Row {
  date: string;
  home: string;
  away: string;
  hg: number;
  ag: number;
  neutral: boolean;
}

function parseCsv(text: string): Row[] {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  const out: Row[] = [];
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i]!.split(",");
    if (c.length < 9) continue;
    const hg = parseInt(c[3]!, 10);
    const ag = parseInt(c[4]!, 10);
    if (!Number.isFinite(hg) || !Number.isFinite(ag)) continue; // NA = jogo futuro
    out.push({
      date: c[0]!,
      home: c[1]!,
      away: c[2]!,
      hg,
      ag,
      neutral: c[8]!.trim().toUpperCase() === "TRUE",
    });
  }
  out.sort((a, b) => a.date.localeCompare(b.date));
  return out;
}

interface MarketAccumulator {
  name: string;
  samples: ForecastSample[];
}

function report(acc: MarketAccumulator): void {
  const n = acc.samples.length;
  const brier = brierScore(acc.samples);
  const base = baseRateBrier(acc.samples);
  const skill = brierSkillScore(acc.samples);
  const rate = n > 0 ? acc.samples.reduce((a, s) => a + s.outcome, 0) / n : 0;
  process.stdout.write(
    `\n## ${acc.name}\n` +
      `  n=${n} base_rate=${rate.toFixed(4)}\n` +
      `  brier_model=${brier?.toFixed(5)} brier_baserate=${base?.toFixed(5)} skill=${skill?.toFixed(4)}\n`,
  );
  const bins = calibrationBins(acc.samples, 10).filter(b => b.count > 0);
  process.stdout.write(`  calibration (pred → empírico, n):\n`);
  for (const b of bins) {
    process.stdout.write(
      `    [${b.lo.toFixed(1)}-${b.hi.toFixed(1)}) pred=${b.meanPredicted?.toFixed(3)} emp=${b.empiricalRate?.toFixed(3)} n=${b.count}\n`,
    );
  }
}

function main(): void {
  if (!CSV_PATH || !fs.existsSync(CSV_PATH)) {
    process.stderr.write(`[backtest] CSV não encontrado em BACKTEST_RESULTS_CSV=${CSV_PATH}\n`);
    process.exit(1);
  }
  const rows = parseCsv(fs.readFileSync(CSV_PATH, "utf8"));
  process.stdout.write(`[backtest] partidas válidas=${rows.length} min_year=${MIN_YEAR} burn_in=${BURN_IN}\n`);

  const elo = new Map<string, number>();
  const games = new Map<string, number>();
  const getElo = (t: string): number => elo.get(t) ?? ELO_INITIAL;
  const getGames = (t: string): number => games.get(t) ?? 0;

  const markets: Record<string, MarketAccumulator> = {
    over05: { name: "Over 0.5 gols", samples: [] },
    over15: { name: "Over 1.5 gols", samples: [] },
    over25: { name: "Over 2.5 gols", samples: [] },
    btts: { name: "Both Teams To Score", samples: [] },
    homeWin: { name: "Resultado: vitória mandante", samples: [] },
  };

  let evaluated = 0;
  for (const r of rows) {
    const year = parseInt(r.date.slice(0, 4), 10);
    const eh = getElo(r.home);
    const ea = getElo(r.away);
    const enoughHistory = getGames(r.home) >= BURN_IN && getGames(r.away) >= BURN_IN;

    if (enoughHistory && year >= MIN_YEAR) {
      const { lambdaHome, lambdaAway } = expectedGoalsFromElo(eh, ea, r.neutral, DEFAULT_ELO_PARAMS);
      const total = r.hg + r.ag;
      const res = matchResultProbs(lambdaHome, lambdaAway);
      markets.over05!.samples.push({ predicted: pOver(lambdaHome, lambdaAway, 0.5), outcome: total >= 1 ? 1 : 0 });
      markets.over15!.samples.push({ predicted: pOver(lambdaHome, lambdaAway, 1.5), outcome: total >= 2 ? 1 : 0 });
      markets.over25!.samples.push({ predicted: pOver(lambdaHome, lambdaAway, 2.5), outcome: total >= 3 ? 1 : 0 });
      markets.btts!.samples.push({ predicted: pBothTeamsScore(lambdaHome, lambdaAway), outcome: r.hg >= 1 && r.ag >= 1 ? 1 : 0 });
      markets.homeWin!.samples.push({ predicted: res.homeWin, outcome: r.hg > r.ag ? 1 : 0 });
      evaluated++;
    }

    const u = updateElo(eh, ea, r.hg, r.ag, r.neutral, DEFAULT_ROLLING_ELO);
    elo.set(r.home, u.home);
    elo.set(r.away, u.away);
    games.set(r.home, getGames(r.home) + 1);
    games.set(r.away, getGames(r.away) + 1);
  }

  process.stdout.write(`[backtest] partidas avaliadas (pós burn-in, ${MIN_YEAR}+)=${evaluated}\n`);
  for (const key of Object.keys(markets)) report(markets[key]!);

  process.stdout.write(
    `\n[backtest] LEITURA: skill > 0 ⇒ modelo bate o preditor de base rate. ` +
      `Calibração boa ⇔ pred ≈ empírico em cada faixa. ` +
      `Isto NÃO prova que bate o mercado — é o portão #1.\n`,
  );
}

main();
