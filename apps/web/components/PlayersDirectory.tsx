"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { CURRENT_SEASON_BANNER, FOOTER } from "@bos/card-schema";
import {
  type PlayerListItem,
  type PlayerSide,
} from "@bos/api-client";

import { SiteHeader } from "@/components/SiteHeader";
import type { PlayersPageData } from "@/lib/load";
import {
  directoryFieldingAdvancedColumns,
  directoryFieldingColumns,
  directoryHittingAdvancedColumns,
  directoryHittingColumns,
  directoryPitchingAdvancedColumns,
  directoryPitchingColumns,
  directoryRowsForSide,
  fieldingAdvancedCells,
  fieldingStandardCells,
  hittingAdvancedCells,
  hittingStandardCells,
  pitchingAdvancedCells,
  pitchingStandardCells,
  primaryFielding,
  statValue,
  type StatCell,
} from "@/lib/playerPage";

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function Chip({
  label,
  selected,
  onClick,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={selected ? "bos-tab is-on" : "bos-tab"}
      aria-pressed={selected}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function DirectoryTable({
  caption,
  rows,
  side,
}: {
  caption: string;
  rows: PlayerListItem[];
  side: PlayerSide;
}) {
  const hitting = rows.map((row) => row.hitting).filter((row): row is NonNullable<typeof row> => Boolean(row));
  const pitching = rows.map((row) => row.pitching).filter((row): row is NonNullable<typeof row> => Boolean(row));
  const fielding = rows.flatMap((row) => row.fielding);
  const standard = side === "pitching"
    ? directoryPitchingColumns(pitching)
    : directoryHittingColumns(hitting);
  const fieldingCols = side === "hitting" ? directoryFieldingColumns(fielding) : [];
  const fieldingAdv = side === "hitting" ? directoryFieldingAdvancedColumns(fielding) : [];
  const advanced = side === "pitching"
    ? directoryPitchingAdvancedColumns(pitching)
    : directoryHittingAdvancedColumns(hitting);

  return (
    <section className="bos-block">
      <h2>{caption}</h2>
      <div className="bos-table-wrap">
        <table className="bos-table bos-table-compact">
          <thead>
            <tr>
              <th>Player</th>
              <th>Pos</th>
              <th>Team</th>
              {standard.map((cell) => (
                <th key={`std-${cell.key}`} className="bos-num">{cell.label}</th>
              ))}
              {fieldingCols.map((cell) => (
                <th key={`fld-${cell.key}`} className={cell.key === "pos" ? undefined : "bos-num"}>
                  {cell.label}
                </th>
              ))}
              {fieldingAdv.map((cell) => (
                <th key={`fadv-${cell.key}`} className="bos-num">{cell.label}</th>
              ))}
              {advanced.map((cell) => (
                <th key={`adv-${cell.key}`} className="bos-num">{cell.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const standardCells: StatCell[] = side === "pitching"
                ? (row.pitching ? pitchingStandardCells(row.pitching) : [])
                : (row.hitting ? hittingStandardCells(row.hitting) : []);
              const fieldLine = primaryFielding(row);
              const fieldCells = fieldLine ? fieldingStandardCells(fieldLine) : [];
              const fieldAdvCells = fieldLine ? fieldingAdvancedCells(fieldLine) : [];
              const advCells = side === "pitching"
                ? (row.pitching ? pitchingAdvancedCells(row.pitching) : [])
                : (row.hitting ? hittingAdvancedCells(row.hitting) : []);
              return (
                <tr key={`${row.player_id}-${row.season}-${side}`}>
                  <td>
                    <Link href={`/players/${encodeURIComponent(row.player_id)}`}>{row.name}</Link>
                  </td>
                  <td>{row.position}</td>
                  <td>{row.team}</td>
                  {standard.map((cell) => (
                    <td key={`std-${cell.key}`} className="bos-num">
                      {statValue(standardCells, cell.key)}
                    </td>
                  ))}
                  {fieldingCols.map((cell) => (
                    <td key={`fld-${cell.key}`} className={cell.key === "pos" ? undefined : "bos-num"}>
                      {statValue(fieldCells, cell.key)}
                    </td>
                  ))}
                  {fieldingAdv.map((cell) => (
                    <td key={`fadv-${cell.key}`} className="bos-num">
                      {statValue(fieldAdvCells, cell.key)}
                    </td>
                  ))}
                  {advanced.map((cell) => (
                    <td key={`adv-${cell.key}`} className="bos-num">
                      {statValue(advCells, cell.key)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function PlayersDirectory({
  bySeason,
  seasons,
  defaultSeason,
  showSeasonBanner,
}: PlayersPageData) {
  const [query, setQuery] = useState("");
  const [season, setSeason] = useState(defaultSeason);
  const [position, setPosition] = useState("");
  const [team, setTeam] = useState("");

  const pool = bySeason[season] ?? [];
  const positions = useMemo(() => uniqueSorted(pool.map((row) => row.position)), [pool]);
  const teams = useMemo(() => uniqueSorted(pool.map((row) => row.team)), [pool]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return pool.filter((row) => {
      if (position && row.position !== position) {
        return false;
      }
      if (team && row.team !== team) {
        return false;
      }
      if (!needle) {
        return true;
      }
      const hay = `${row.name} ${row.team} ${row.position} ${row.player_id}`.toLowerCase();
      return hay.includes(needle);
    });
  }, [pool, position, query, team]);

  const battingRows = useMemo(() => directoryRowsForSide(rows, "hitting"), [rows]);
  const pitchingRows = useMemo(() => directoryRowsForSide(rows, "pitching"), [rows]);

  return (
    <div className="bos-shell bos-shell-wide">
      <SiteHeader active="players" />

      {showSeasonBanner ? (
        <p className="bos-banner" role="status">
          {CURRENT_SEASON_BANNER}
        </p>
      ) : null}

      <section className="bos-pagehead">
        <h1>Players</h1>
        <p>Season batting, pitching, and fielding lines for the live window, sorted by WAR.</p>
      </section>

      <div className="bos-filters">
        <label className="bos-search">
          <span className="bos-sr">Search players</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search name, team, or position"
          />
        </label>
        <div className="bos-tabs" role="tablist" aria-label="Season">
          {seasons.map((year) => (
            <button
              key={year}
              type="button"
              role="tab"
              aria-selected={season === year}
              className={season === year ? "bos-tab is-on" : "bos-tab"}
              onClick={() => {
                setSeason(year);
                setPosition("");
                setTeam("");
              }}
            >
              {year}
            </button>
          ))}
        </div>
        <div className="bos-chiprow" role="group" aria-label="Position">
          <Chip label="All pos" selected={!position} onClick={() => setPosition("")} />
          {positions.map((value) => (
            <Chip
              key={value}
              label={value}
              selected={position === value}
              onClick={() => setPosition(value)}
            />
          ))}
        </div>
        <div className="bos-chiprow" role="group" aria-label="Team">
          <Chip label="All teams" selected={!team} onClick={() => setTeam("")} />
          {teams.map((value) => (
            <Chip
              key={value}
              label={value}
              selected={team === value}
              onClick={() => setTeam(value)}
            />
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="bos-empty" role="status">
          <h2>No players in this slice</h2>
          <p>Try another season or clear the filters. We do not invent missing-year rows.</p>
        </div>
      ) : (
        <>
          {battingRows.length ? (
            <DirectoryTable caption="Batting" rows={battingRows} side="hitting" />
          ) : null}
          {pitchingRows.length ? (
            <DirectoryTable caption="Pitching" rows={pitchingRows} side="pitching" />
          ) : null}
        </>
      )}

      <footer className="bos-foot">{FOOTER}</footer>
    </div>
  );
}
