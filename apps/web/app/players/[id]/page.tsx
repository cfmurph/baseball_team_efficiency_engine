import type { Metadata } from "next";

import { PRODUCT_NAME } from "@bos/card-schema";

import { PlayerProfile } from "@/components/PlayerProfile";
import { loadPlayerData } from "@/lib/load";

export const dynamic = "force-dynamic";

type Params = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Params): Promise<Metadata> {
  const { id } = await params;
  const data = await loadPlayerData(id);
  const name = data.detail?.player.name;
  return {
    title: name ? `${name} · ${PRODUCT_NAME}` : PRODUCT_NAME,
  };
}

export default async function PlayerPage({ params }: Params) {
  const { id } = await params;
  const data = await loadPlayerData(id);
  return <PlayerProfile {...data} />;
}
