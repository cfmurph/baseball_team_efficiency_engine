import { PlayersDirectory } from "@/components/PlayersDirectory";
import { loadPlayersData } from "@/lib/load";

export const dynamic = "force-dynamic";

export default async function PlayersPage() {
  const data = await loadPlayersData();
  return <PlayersDirectory {...data} />;
}
