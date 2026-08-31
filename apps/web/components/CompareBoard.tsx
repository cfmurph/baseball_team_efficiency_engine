"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { selectedYearMissing, type PlayerListItem } from "@bos/api-client";
import { CURRENT_SEASON_BANNER, FOOTER } from "@bos/card-schema";

import { SiteHeader } from "@/components/SiteHeader";
import {
  COMPARE_MIN,
  appendCompareId,
  buildComparePath,
  buildCompareRows,
  columnFromDetail,
  filterSlotCandidates,
  removeCompareId,
  slotIds,
  writeStoredCompare,
  type CompareRowView,
} from "@/lib/compare";
import type { ComparePageData } from "@/lib/load";

function CompareStatRows({
  row,
  columns,
  showSection,
}: {
  row: CompareRowView;
  columns: number;
  showSection: boolean;
}) {
  return (
    <>
      {showSection ? (
        <tr className="bos-compare-section-row">
          <th scope="colgroup" colSpan={columns + 1}>
            {row.block === "hitting" ? "Hitting" : "Pitching"}
          </th>
        </tr>
      ) : null}
      <tr>
        <th scope="row">{row.label}</th>
        {row.display.map((value, columnIndex) => (
          <td
            key={`${row.key}-${columnIndex}`}
            className={row.best.includes(columnIndex) ? "bos-num is-best" : "bos-num"}
          >
            {value}
          </td>
        ))}
      </tr>
    </>
  );
}

function PlayerSlot({
  index,
  playerId,
  column,
  pool,
  selectedIds,
  onAdd,
  onRemove,
}: {
  index: number;
  playerId: string | null;
  column: ReturnType<typeof columnFromDetail> | null;
  pool: PlayerListItem[];
  selectedIds: string[];
  onAdd: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const matches = useMemo(
    () => filterSlotCandidates(pool, query, selectedIds),
    [pool, query, selectedIds],
  );

  if (playerId && column) {
    return (
      <div className="bos-slot is-filled">
        <p className="bos-slot-name">
          {column.found ? (
            <Link href={`/players/${encodeURIComponent(column.player_id)}`}>{column.name}</Link>
          ) : (
            column.name
          )}
        </p>
        <p className="bos-slot-meta">
          {[column.position, column.team].filter(Boolean).join(" · ") || "No identity yet"}
        </p>
        <button
          type="button"
          className="bos-slot-remove"
          onClick={() => onRemove(playerId)}
          aria-label={`Remove ${column.name}`}
        >
          Remove
        </button>
      </div>
    );
  }

  return (
    <div className="bos-slot">
      <label className="bos-search">
        <span className="bos-sr">{`Add player ${index + 1}`}</span>
        <input
          type="search"
          role="combobox"
          aria-expanded={open && matches.length > 0}
          aria-controls={`bos-slot-list-${index}`}
          aria-autocomplete="list"
          value={query}
          placeholder="Name, team, or pos"
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => {
            window.setTimeout(() => setOpen(false), 120);
          }}
        />
      </label>
      {open && matches.length > 0 ? (
        <ul className="bos-slot-list" role="listbox" id={`bos-slot-list-${index}`}>
          {matches.map((row) => (
            <li key={row.player_id} role="option">
              <button
                type="button"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  onAdd(row.player_id);
                  setQuery("");
                  setOpen(false);
                }}
              >
                <span>{row.name}</span>
                <span className="bos-slot-meta">
                  {row.position} · {row.team}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function CompareBoard({
  query,
  details,
  bySeason,
  seasons,
  health,
  showSeasonBanner,
}: ComparePageData) {
  const router = useRouter();
  const pool = bySeason[query.season] ?? [];
  const columns = useMemo(
    () =>
      query.ids.map((id, index) => {
        const fallback = pool.find((row) => row.player_id === id) ?? null;
        return columnFromDetail(id, details[index] ?? null, query.season, fallback);
      }),
    [details, pool, query.ids, query.season],
  );
  const rows = useMemo(
    () => (query.ids.length >= COMPARE_MIN ? buildCompareRows(columns) : []),
    [columns, query.ids.length],
  );
  const slots = slotIds(query.ids);
  const hasBoard = query.ids.length >= COMPARE_MIN;
  const hasAnyLine = columns.some((column) => column.hitting || column.pitching);
  const yearMissing = selectedYearMissing(health, query.season, hasAnyLine);
  const showBanner = showSeasonBanner || yearMissing;
  const hasBothBlocks = rows.some((row) => row.block === "hitting") && rows.some((row) => row.block === "pitching");
  const ranked = filterSlotCandidates(pool, "", query.ids, 8);

  useEffect(() => {
    writeStoredCompare({ season: query.season, ids: query.ids });
  }, [query.ids, query.season]);

  function go(ids: string[], season = query.season) {
    writeStoredCompare({ season, ids });
    router.replace(buildComparePath({ season, ids }), { scroll: false });
  }

  return (
    <div className="bos-shell bos-shell-wide bos-shell-compare">
      <SiteHeader active="compare" />

      {showBanner ? (
        <p className="bos-banner" role="status">
          {CURRENT_SEASON_BANNER}
        </p>
      ) : null}

      <section className="bos-pagehead">
        <h1>Compare</h1>
        <p>Two to four players, one season, side by side.</p>
      </section>

      <div className="bos-tabs" role="tablist" aria-label="Compare mode">
        <button type="button" role="tab" aria-selected="true" className="bos-tab is-on">
          Players
        </button>
        <button
          type="button"
          role="tab"
          aria-selected="false"
          className="bos-tab"
          disabled
          title="Team compare comes next"
        >
          Teams
        </button>
      </div>

      <div className="bos-tabs" role="tablist" aria-label="Season">
        {seasons.map((year) => (
          <button
            key={year}
            type="button"
            role="tab"
            aria-selected={query.season === year}
            className={query.season === year ? "bos-tab is-on" : "bos-tab"}
            onClick={() => go(query.ids, year)}
          >
            {year}
          </button>
        ))}
      </div>

      <div className="bos-slots">
        {slots.map((playerId, index) => (
          <PlayerSlot
            key={playerId ? `${playerId}-${index}` : `empty-${index}`}
            index={index}
            playerId={playerId}
            column={playerId ? columns[query.ids.indexOf(playerId)] ?? null : null}
            pool={pool}
            selectedIds={query.ids}
            onAdd={(id) => go(appendCompareId(query.ids, id))}
            onRemove={(id) => go(removeCompareId(query.ids, id))}
          />
        ))}
      </div>

      {!hasBoard ? (
        <div className="bos-empty" role="status">
          <h2>{query.ids.length === 1 ? "Add one more player" : "Add two players to compare"}</h2>
          <p>
            {query.ids.length === 1
              ? "The board needs at least two names. Search a slot or pick from this season."
              : "Four slots. Search a name, team, or position — we do not invent missing-year rows."}
          </p>
        </div>
      ) : null}

      {!hasBoard && ranked.length > 0 ? (
        <section className="bos-block">
          <h2>Add from this season</h2>
          <ul className="bos-picklist">
            {ranked.map((row) => (
              <li key={row.player_id}>
                <button type="button" className="bos-ghost" onClick={() => go(appendCompareId(query.ids, row.player_id))}>
                  <span>{row.name}</span>
                  <span className="bos-slot-meta">
                    {row.position} · {row.team}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {hasBoard && !hasAnyLine ? (
        <div className="bos-empty" role="status">
          <h2>No {query.season} line yet</h2>
          <p>Empty board until the nightly publishes that year. We do not invent rows.</p>
        </div>
      ) : null}

      {hasBoard && hasAnyLine ? (
        <div className="bos-table-wrap bos-compare-wrap">
          <table className="bos-table bos-compare-table">
            <thead>
              <tr>
                <th scope="col">Stat</th>
                {columns.map((column) => (
                  <th key={column.player_id} scope="col">
                    <div className="bos-compare-id">
                      {column.found ? (
                        <Link href={`/players/${encodeURIComponent(column.player_id)}`}>{column.name}</Link>
                      ) : (
                        <span>{column.name}</span>
                      )}
                      <span className="bos-slot-meta">
                        {[column.position, column.team].filter(Boolean).join(" · ")}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => {
                const showSection = hasBothBlocks && row.block !== rows[index - 1]?.block;
                return (
                  <CompareStatRows
                    key={row.key}
                    row={row}
                    columns={columns.length}
                    showSection={showSection}
                  />
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}

      <footer className="bos-foot">{FOOTER}</footer>
    </div>
  );
}
